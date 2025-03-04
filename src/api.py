"""API Interface for AI-OPS"""
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import API_SETTINGS
from src.routers import session_router, rag_router
from src.utils import get_logger
from src.routers import session_router, rag_router

logger = get_logger(__name__)

# Initialize API
app = FastAPI()
app.include_router(session_router)
app.include_router(rag_router)  # Add the RAG router

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_SETTINGS.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring if enabled
if API_SETTINGS.PROFILE:
    try:
        from src.routers.monitoring import monitor_router
        app.mount('/monitor', monitor_router)
    except RuntimeError as monitor_startup_err:
        logger.error("Monitoring disabled: ", str(monitor_startup_err))

@app.get('/ping')
def ping():
    """Used to check if API is on"""
    return status.HTTP_200_OK