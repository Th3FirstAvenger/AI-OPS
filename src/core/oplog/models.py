"""Data models for Red Team Operation Logging"""
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions that can be logged"""
    COMMAND = "command"        # CLI command execution
    RDP = "rdp"               # RDP session
    GUI_TOOL = "gui_tool"     # GUI tool usage (Burp, etc.)
    MANUAL = "manual"         # Manual action
    NOTE = "note"             # General note
    FILE_TRANSFER = "file_transfer"
    EXPLOIT = "exploit"
    CREDENTIAL = "credential"
    PERSISTENCE = "persistence"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privesc"
    EXFILTRATION = "exfil"


class Phase(str, Enum):
    """Phases of penetration testing"""
    RECON = "recon"
    SCANNING = "scanning"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    PERSISTENCE = "persistence"
    LATERAL_MOVEMENT = "lateral_movement"
    EXFILTRATION = "exfiltration"
    CLEANUP = "cleanup"


class Target(BaseModel):
    """Target system information"""
    id: Optional[int] = None
    name: str = Field(..., description="Target identifier (hostname, IP, etc.)")
    ip_address: Optional[str] = None
    os: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Operation(BaseModel):
    """Penetration testing operation/engagement"""
    id: Optional[int] = None
    name: str = Field(..., description="Operation/Engagement name")
    client: Optional[str] = None
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    description: Optional[str] = None
    is_active: bool = True


class LogEntry(BaseModel):
    """Individual operation log entry"""
    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Context
    operator: str = Field(..., description="Username of operator")
    hostname: str = Field(..., description="Hostname of operator's machine")
    operation_id: Optional[int] = None
    target_id: Optional[int] = None
    phase: Optional[Phase] = None

    # Action details
    action_type: ActionType
    command: Optional[str] = None
    tool_name: Optional[str] = None
    description: str = Field(..., description="Human-readable description")

    # Output/Results
    output: Optional[str] = None
    success: Optional[bool] = None

    # Metadata
    tags: List[str] = Field(default_factory=list)
    sensitive: bool = Field(default=False, description="Contains sensitive data (credentials, etc.)")

    # Sync status
    synced: bool = False
    sync_timestamp: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OperationContext(BaseModel):
    """Current operation context for the CLI"""
    operation: Optional[Operation] = None
    current_target: Optional[Target] = None
    current_phase: Optional[Phase] = None
    auto_log: bool = True  # Auto-log commands by default
    log_filter: List[str] = Field(
        default_factory=lambda: ["ls", "cd", "pwd", "clear", "exit", "history"]
    )  # Commands to NOT auto-log
