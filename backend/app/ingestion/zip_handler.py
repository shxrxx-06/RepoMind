import shutil
import stat
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from backend.app.core.config import settings

MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_ZIP_ENTRIES = 10_000


def extract_repository_zip(zip_path: Path) -> Path:
    """Safely extract an archive into a unique repository directory."""
    destination = settings.REPOSITORIES_DIR / str(uuid.uuid4())
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ZIP_ENTRIES:
                raise ValueError("ZIP contains too many entries")
            if sum(item.file_size for item in entries) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("ZIP exceeds extraction size limit")

            root = destination.resolve()
            extracted_size = 0
            for item in entries:
                name = item.filename.replace("\\", "/")
                member = PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts or "\x00" in name:
                    raise ValueError("ZIP contains an unsafe path")
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("ZIP contains a symbolic link")
                target = (destination / Path(*member.parts)).resolve()
                if target != root and root not in target.parents:
                    raise ValueError("ZIP contains an unsafe path")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with archive.open(item, "r") as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        extracted_size += len(chunk)
                        if (written > item.file_size
                                or extracted_size > MAX_UNCOMPRESSED_BYTES):
                            raise ValueError("ZIP entry exceeded its declared size")
                        output.write(chunk)
        return destination
    except (zipfile.BadZipFile, OSError, RuntimeError, ValueError):
        shutil.rmtree(destination, ignore_errors=True)
        raise
