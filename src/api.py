"""API Interface for AI-OPS"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import API_SETTINGS
from src.routers import session_router, rag_router
from src.utils import get_logger

logger = get_logger(__name__)

# Initialize API con descripción para OpenAPI
app = FastAPI(
    title="AI-OPS API",
    description="Interfaz de API para el sistema AI-OPS.",
    version="1.0.0"
)
app.include_router(session_router)
app.include_router(rag_router)

# CORS middleware con métodos y headers específicos
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_SETTINGS.ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Ajusta según necesites
    allow_headers=["Content-Type", "Authorization"],  # Ajusta según necesites
)

# Monitoring if enabled
if API_SETTINGS.PROFILE:
    try:
        from src.routers.monitoring import monitor_router
        app.mount('/monitor', monitor_router)
    except RuntimeError as monitor_startup_err:
        logger.error(f"Monitoring disabled: {str(monitor_startup_err)}")

@app.get('/ping', summary="Verifica si la API está activa")
def ping():
    """Used to check if API is on"""
    return {"status": "ok"}