"""API Router for Red Team Operation Logging"""
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.core.oplog import OperationLog, LogEntry, Operation, Target
from src.core.oplog.models import ActionType, Phase
from src.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/oplog", tags=["oplog"])


# ==================== REQUEST/RESPONSE MODELS ====================

class LogEntrySync(BaseModel):
    """Log entry for synchronization (without ID)"""
    timestamp: datetime
    operator: str
    hostname: str
    operation_id: Optional[int] = None
    target_id: Optional[int] = None
    phase: Optional[str] = None
    action_type: str
    command: Optional[str] = None
    tool_name: Optional[str] = None
    description: str
    output: Optional[str] = None
    success: Optional[bool] = None
    tags: List[str] = []
    sensitive: bool = False


class SyncRequest(BaseModel):
    """Request to sync multiple log entries"""
    logs: List[LogEntrySync]


class SyncResponse(BaseModel):
    """Response after syncing logs"""
    success: bool
    synced_count: int
    message: str


class OperationResponse(BaseModel):
    """Response with operation details"""
    id: int
    name: str
    client: Optional[str]
    start_date: datetime
    end_date: Optional[datetime]
    description: Optional[str]
    is_active: bool


class StatsResponse(BaseModel):
    """Statistics response"""
    total_logs: int
    total_operations: int
    total_targets: int
    by_operator: dict
    by_action_type: dict
    recent_activity: List[dict]


# ==================== ENDPOINTS ====================

# Initialize OpLog for server
# This creates a separate database for the central server
server_oplog = OperationLog()


@router.post("/sync", response_model=SyncResponse)
async def sync_logs(request: SyncRequest):
    """
    Synchronize logs from client to server.

    Receives logs from operators and stores them in the central database.
    """
    try:
        synced_count = 0

        for log in request.logs:
            # Convert phase string to Phase enum if present
            phase = None
            if log.phase:
                try:
                    phase = Phase(log.phase)
                except ValueError:
                    logger.warning(f"Invalid phase value: {log.phase}")

            # Convert action_type string to ActionType enum
            try:
                action_type = ActionType(log.action_type)
            except ValueError:
                logger.warning(f"Invalid action_type: {log.action_type}")
                action_type = ActionType.MANUAL

            # Add log to server database
            server_oplog.add_log(
                action_type=action_type,
                description=log.description,
                command=log.command,
                tool_name=log.tool_name,
                output=log.output,
                success=log.success,
                operation_id=log.operation_id,
                target_id=log.target_id,
                phase=phase,
                tags=log.tags,
                sensitive=log.sensitive
            )
            synced_count += 1

        logger.info(f"Synced {synced_count} logs from {request.logs[0].operator if request.logs else 'unknown'}")

        return SyncResponse(
            success=True,
            synced_count=synced_count,
            message=f"Successfully synced {synced_count} log entries"
        )

    except Exception as e:
        logger.error(f"Error syncing logs: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/logs")
async def get_logs(
    operation_id: Optional[int] = Query(None, description="Filter by operation ID"),
    operator: Optional[str] = Query(None, description="Filter by operator"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return")
):
    """
    Get consolidated logs from the central database.

    Returns logs from all operators, optionally filtered.
    """
    try:
        logs = server_oplog.get_logs(
            operation_id=operation_id,
            limit=limit
        )

        # Filter by operator if specified
        if operator:
            logs = [log for log in logs if log.operator == operator]

        return {
            "success": True,
            "count": len(logs),
            "logs": [log.model_dump() for log in logs]
        }

    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.get("/operations", response_model=List[OperationResponse])
async def list_operations():
    """
    List all operations in the central database.
    """
    try:
        operations = server_oplog.list_operations()
        return [
            OperationResponse(
                id=op.id,
                name=op.name,
                client=op.client,
                start_date=op.start_date,
                end_date=op.end_date,
                description=op.description,
                is_active=op.is_active
            )
            for op in operations
        ]

    except Exception as e:
        logger.error(f"Error listing operations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list operations: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    operation_id: Optional[int] = Query(None, description="Filter by operation ID")
):
    """
    Get statistics from the central database.
    """
    try:
        # Get basic stats
        stats = server_oplog.get_stats(operation_id=operation_id)

        # Get all logs for additional analysis
        all_logs = server_oplog.get_logs(operation_id=operation_id, limit=10000)

        # Count by operator
        by_operator = {}
        for log in all_logs:
            by_operator[log.operator] = by_operator.get(log.operator, 0) + 1

        # Recent activity (last 10 logs)
        recent_logs = server_oplog.get_logs(operation_id=operation_id, limit=10)
        recent_activity = [
            {
                "timestamp": log.timestamp.isoformat(),
                "operator": log.operator,
                "action_type": log.action_type.value,
                "description": log.description
            }
            for log in recent_logs
        ]

        # Get operation and target counts
        operations = server_oplog.list_operations()
        targets = server_oplog.list_targets()

        return StatsResponse(
            total_logs=stats['total'],
            total_operations=len(operations),
            total_targets=len(targets),
            by_operator=by_operator,
            by_action_type=stats['by_type'],
            recent_activity=recent_activity
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/operations", response_model=OperationResponse)
async def create_operation(
    name: str,
    client: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Create a new operation in the central database.
    """
    try:
        operation = Operation(
            name=name,
            client=client,
            description=description,
            is_active=True
        )

        op_id = server_oplog.create_operation(operation)
        operation.id = op_id

        logger.info(f"Created operation: {name} (ID: {op_id})")

        return OperationResponse(
            id=operation.id,
            name=operation.name,
            client=operation.client,
            start_date=operation.start_date,
            end_date=operation.end_date,
            description=operation.description,
            is_active=operation.is_active
        )

    except Exception as e:
        logger.error(f"Error creating operation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create operation: {str(e)}")


@router.get("/targets")
async def list_targets():
    """
    List all targets in the central database.
    """
    try:
        targets = server_oplog.list_targets()
        return {
            "success": True,
            "count": len(targets),
            "targets": [target.model_dump() for target in targets]
        }

    except Exception as e:
        logger.error(f"Error listing targets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list targets: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for OpLog API.
    """
    return {
        "status": "healthy",
        "service": "oplog",
        "timestamp": datetime.utcnow().isoformat()
    }
