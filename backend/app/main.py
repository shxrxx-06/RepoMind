from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered Codebase Intelligence Platform",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "RepoMind",
        "description": "Repository Intelligence Platform",
        "version": "0.1.0",
        "docs": "/docs",
    }
