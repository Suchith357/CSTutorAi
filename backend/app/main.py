from fastapi import FastAPI

from backend.app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    description="An interactive, personalized and source-grounded Computer Science AI tutor.",
    version=settings.app_version,
)


@app.get("/")
def root():
    return {
        "project": settings.app_name,
        "status": "running",
        "version": settings.app_version,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
    }