function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return null;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function FileRow({ file }) {
  return (
    <li className="file-row">
      <span className="file-type-mark" aria-hidden="true">{file.path.split(".").pop()?.slice(0, 2).toUpperCase() || "--"}</span>
      <span className="file-path" title={file.path}>{file.path}</span>
      <span className="file-language">{file.language || "Other"}</span>
      <span className="file-size">{formatBytes(file.size_bytes) || "—"}</span>
    </li>
  );
}

export default function RepositoryOverview({ repository }) {
  const metadata = repository.metadata;
  const languages = Object.entries(metadata.language_distribution || {}).sort(
    ([left], [right]) => left.localeCompare(right),
  );
  const files = Array.isArray(metadata.files) ? metadata.files : [];
  const displayedFiles = files.slice(0, 8);

  return (
    <section className="overview-card" aria-labelledby="overview-title">
      <div className="overview-topline">
        <div>
          <span className="eyebrow">INGESTION COMPLETE</span>
          <h2 id="overview-title">Repository overview</h2>
        </div>
        <span className="status-pill"><span /> Ready</span>
      </div>
      <p className="success-message">{repository.message}</p>
      <div className="repo-id-row">
        <span>Repository ID</span>
        <code>{repository.repository_id}</code>
      </div>

      <div className="stat-grid" aria-label="Repository scan summary">
        <div className="stat-tile">
          <span className="stat-label">Scanned files</span>
          <strong>{metadata.total_files}</strong>
        </div>
        <div className="stat-tile">
          <span className="stat-label">Repository size</span>
          <strong>{formatBytes(metadata.total_size_bytes) || "—"}</strong>
        </div>
        <div className="stat-tile">
          <span className="stat-label">Skipped files</span>
          <strong>{metadata.skipped_files}</strong>
        </div>
      </div>

      <div className="overview-section">
        <div className="section-title-row">
          <h3>Languages detected</h3>
          <span>{languages.length} {languages.length === 1 ? "language" : "languages"}</span>
        </div>
        {languages.length ? (
          <ul className="language-list">
            {languages.map(([name, count]) => (
              <li className="language-chip" key={name}>
                <span className="language-dot" />
                <span>{name}</span>
                <strong>{count}</strong>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted-empty">No recognized languages were reported for this repository.</p>
        )}
      </div>

      <div className="overview-section files-section">
        <div className="section-title-row">
          <h3>Files</h3>
          <span>{files.length} returned</span>
        </div>
        {displayedFiles.length ? (
          <ul className="file-list">
            {displayedFiles.map((file) => <FileRow key={file.path} file={file} />)}
          </ul>
        ) : (
          <p className="muted-empty">No file entries were returned.</p>
        )}
        {files.length > displayedFiles.length && (
          <p className="showing-note">Showing {displayedFiles.length} of {files.length} files returned by the scanner.</p>
        )}
      </div>
    </section>
  );
}
