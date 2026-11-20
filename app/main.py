from fastapi import FastAPI
from app.core.config import settings
from app.api.v1 import api_router

app = FastAPI(
    
    title=settings.PROJECT_NAME,
    description="A secure, on-premise multimodal AI workbench.",
    version="0.1.0",
)
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def home():
    return {
        "project": settings.PROJECT_NAME,
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
    }