"""
API Interface for AI-OPS, includes Sessions routes, Collections routes, and OpLog routes:

- **Sessions**: Agent related operations including chat and conversation management.

- **Collections**: RAG related operations (...)

- **OpLog**: Red Team Operation Logging API for centralized log synchronization.

### RAG Routes
- /collections/list    : Returns available Collections.
- /collections/new     : Creates a new Collection.
- /collections/upload/ : Upload document to an existing Collection

### OpLog Routes
- /oplog/sync          : Sync logs from operators to central server
- /oplog/logs          : Get consolidated logs
- /oplog/operations    : List all operations
- /oplog/stats         : Get operation statistics
- /oplog/health        : Health check
"""
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import API_SETTINGS
from src.routers import session_router
from src.routers.oplog import router as oplog_router
from src.utils import get_logger

logger = get_logger(__name__)

# --- Initialize API
app = FastAPI()
app.include_router(session_router)
app.include_router(oplog_router)

# TODO: implement proper CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_SETTINGS.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

