"""SQLite database for operation logging with offline-first design"""
import sqlite3
import socket
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from contextlib import contextmanager
import requests

from src.core.oplog.models import LogEntry, Operation, Target, ActionType, Phase
from src.utils import get_logger

logger = get_logger(__name__)

# Database path
OPLOG_PATH = Path(Path.home() / '.aiops' / 'oplog')
if not OPLOG_PATH.exists():
    OPLOG_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created {str(OPLOG_PATH)}")

DB_PATH = OPLOG_PATH / 'operations.db'


class OperationLog:
    """
    Manages operation logging with SQLite database.
    Offline-first design: logs are stored locally and synced later.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()
        self.operator = os.getenv('USER', 'unknown')
        self.hostname = socket.gethostname()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Operations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    client TEXT,
                    start_date TEXT NOT NULL,
                    end_date TEXT,
                    description TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Targets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    ip_address TEXT,
                    os TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Log entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS log_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operator TEXT NOT NULL,
                    hostname TEXT NOT NULL,
                    operation_id INTEGER,
                    target_id INTEGER,
                    phase TEXT,
                    action_type TEXT NOT NULL,
                    command TEXT,
                    tool_name TEXT,
                    description TEXT NOT NULL,
                    output TEXT,
                    success INTEGER,
                    tags TEXT,
                    sensitive INTEGER DEFAULT 0,
                    synced INTEGER DEFAULT 0,
                    sync_timestamp TEXT,
                    FOREIGN KEY (operation_id) REFERENCES operations(id),
                    FOREIGN KEY (target_id) REFERENCES targets(id)
                )
            """)

            # Indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_timestamp
                ON log_entries(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_operation
                ON log_entries(operation_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_synced
                ON log_entries(synced)
            """)

            conn.commit()
            logger.info("Database initialized")

    # === OPERATIONS ===

    def create_operation(self, operation: Operation) -> int:
        """Create a new operation/engagement"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO operations (name, client, start_date, description, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (
                operation.name,
                operation.client,
                operation.start_date.isoformat(),
                operation.description,
                1 if operation.is_active else 0
            ))
            conn.commit()
            operation_id = cursor.lastrowid
            logger.info(f"Created operation: {operation.name} (ID: {operation_id})")
            return operation_id

    def get_operation(self, operation_id: int) -> Optional[Operation]:
        """Get operation by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM operations WHERE id = ?", (operation_id,))
            row = cursor.fetchone()
            if row:
                return Operation(
                    id=row['id'],
                    name=row['name'],
                    client=row['client'],
                    start_date=datetime.fromisoformat(row['start_date']),
                    end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
                    description=row['description'],
                    is_active=bool(row['is_active'])
                )
            return None

    def get_active_operation(self) -> Optional[Operation]:
        """Get the currently active operation"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM operations
                WHERE is_active = 1
                ORDER BY start_date DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return Operation(
                    id=row['id'],
                    name=row['name'],
                    client=row['client'],
                    start_date=datetime.fromisoformat(row['start_date']),
                    end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
                    description=row['description'],
                    is_active=bool(row['is_active'])
                )
            return None

    def list_operations(self) -> List[Operation]:
        """List all operations"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM operations ORDER BY start_date DESC")
            rows = cursor.fetchall()
            return [
                Operation(
                    id=row['id'],
                    name=row['name'],
                    client=row['client'],
                    start_date=datetime.fromisoformat(row['start_date']),
                    end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
                    description=row['description'],
                    is_active=bool(row['is_active'])
                )
                for row in rows
            ]

    def set_active_operation(self, operation_id: int):
        """Set an operation as active (deactivates others)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Deactivate all
            cursor.execute("UPDATE operations SET is_active = 0")
            # Activate selected
            cursor.execute("UPDATE operations SET is_active = 1 WHERE id = ?", (operation_id,))
            conn.commit()
            logger.info(f"Set active operation: {operation_id}")

    def delete_operation(self, operation_id: int) -> bool:
        """Delete an operation and all associated logs and targets"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Delete associated log entries
                cursor.execute("DELETE FROM log_entries WHERE operation_id = ?", (operation_id,))
                # Delete associated targets
                cursor.execute("DELETE FROM targets WHERE operation_id = ?", (operation_id,))
                # Delete operation
                cursor.execute("DELETE FROM operations WHERE id = ?", (operation_id,))
                conn.commit()
                logger.info(f"Deleted operation ID: {operation_id} and all associated data")
                return True
            except Exception as e:
                logger.error(f"Failed to delete operation {operation_id}: {e}")
                return False

    # === TARGETS ===

    def create_target(self, target: Target) -> int:
        """Create a new target"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO targets (name, ip_address, os, notes)
                VALUES (?, ?, ?, ?)
            """, (target.name, target.ip_address, target.os, target.notes))
            conn.commit()
            target_id = cursor.lastrowid
            logger.info(f"Created target: {target.name} (ID: {target_id})")
            return target_id

    def get_target_by_name(self, name: str) -> Optional[Target]:
        """Get target by name"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM targets WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return Target(
                    id=row['id'],
                    name=row['name'],
                    ip_address=row['ip_address'],
                    os=row['os'],
                    notes=row['notes'],
                    created_at=datetime.fromisoformat(row['created_at'])
                )
            return None

    def list_targets(self) -> List[Target]:
        """List all targets"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM targets ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [
                Target(
                    id=row['id'],
                    name=row['name'],
                    ip_address=row['ip_address'],
                    os=row['os'],
                    notes=row['notes'],
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                for row in rows
            ]

    # === LOG ENTRIES ===

    def add_log(
        self,
        action_type: ActionType,
        description: str,
        command: Optional[str] = None,
        tool_name: Optional[str] = None,
        output: Optional[str] = None,
        success: Optional[bool] = None,
        operation_id: Optional[int] = None,
        target_id: Optional[int] = None,
        phase: Optional[Phase] = None,
        tags: Optional[List[str]] = None,
        sensitive: bool = False
    ) -> int:
        """Add a log entry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            tags_str = ','.join(tags) if tags else ''

            cursor.execute("""
                INSERT INTO log_entries (
                    timestamp, operator, hostname, operation_id, target_id,
                    phase, action_type, command, tool_name, description,
                    output, success, tags, sensitive
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                self.operator,
                self.hostname,
                operation_id,
                target_id,
                phase.value if phase else None,
                action_type.value,
                command,
                tool_name,
                description,
                output,
                1 if success else (0 if success is False else None),
                tags_str,
                1 if sensitive else 0
            ))
            conn.commit()
            log_id = cursor.lastrowid
            return log_id

    def get_logs(
        self,
        operation_id: Optional[int] = None,
        target_id: Optional[int] = None,
        limit: int = 100,
        unsynced_only: bool = False
    ) -> List[LogEntry]:
        """Get log entries with optional filters"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM log_entries WHERE 1=1"
            params = []

            if operation_id:
                query += " AND operation_id = ?"
                params.append(operation_id)

            if target_id:
                query += " AND target_id = ?"
                params.append(target_id)

            if unsynced_only:
                query += " AND synced = 0"

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                LogEntry(
                    id=row['id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    operator=row['operator'],
                    hostname=row['hostname'],
                    operation_id=row['operation_id'],
                    target_id=row['target_id'],
                    phase=Phase(row['phase']) if row['phase'] else None,
                    action_type=ActionType(row['action_type']),
                    command=row['command'],
                    tool_name=row['tool_name'],
                    description=row['description'],
                    output=row['output'],
                    success=bool(row['success']) if row['success'] is not None else None,
                    tags=row['tags'].split(',') if row['tags'] else [],
                    sensitive=bool(row['sensitive']),
                    synced=bool(row['synced']),
                    sync_timestamp=datetime.fromisoformat(row['sync_timestamp']) if row['sync_timestamp'] else None
                )
                for row in rows
            ]

    def mark_synced(self, log_ids: List[int]):
        """Mark log entries as synced"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(log_ids))
            cursor.execute(f"""
                UPDATE log_entries
                SET synced = 1, sync_timestamp = ?
                WHERE id IN ({placeholders})
            """, [datetime.utcnow().isoformat()] + log_ids)
            conn.commit()
            logger.info(f"Marked {len(log_ids)} logs as synced")

    def get_stats(self, operation_id: Optional[int] = None) -> dict:
        """Get statistics about logs"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            base_query = "SELECT COUNT(*) as count FROM log_entries"
            where_clause = ""
            params = []

            if operation_id:
                where_clause = " WHERE operation_id = ?"
                params = [operation_id]

            # Total logs
            cursor.execute(f"{base_query}{where_clause}", params)
            total = cursor.fetchone()['count']

            # By action type
            cursor.execute(f"""
                SELECT action_type, COUNT(*) as count
                FROM log_entries{where_clause}
                GROUP BY action_type
            """, params)
            by_type = {row['action_type']: row['count'] for row in cursor.fetchall()}

            # Unsynced
            unsynced_query = f"{base_query}{where_clause}"
            if where_clause:
                unsynced_query += " AND synced = 0"
            else:
                unsynced_query += " WHERE synced = 0"

            cursor.execute(unsynced_query, params)
            unsynced = cursor.fetchone()['count']

            return {
                'total': total,
                'by_type': by_type,
                'unsynced': unsynced
            }

    # === SYNCHRONIZATION ===

    def sync_to_server(self, server_url: str, timeout: int = 10) -> dict:
        """
        Synchronize unsynced logs to central server.

        Args:
            server_url: Base URL of the server (e.g., 'http://127.0.0.1:8000')
            timeout: Request timeout in seconds

        Returns:
            dict with 'success', 'synced_count', 'message', and 'error' (if failed)
        """
        try:
            # Get unsynced logs
            unsynced_logs = self.get_logs(unsynced_only=True, limit=10000)

            if not unsynced_logs:
                return {
                    'success': True,
                    'synced_count': 0,
                    'message': 'No logs to sync'
                }

            # Prepare payload
            logs_data = []
            for log in unsynced_logs:
                log_dict = {
                    'timestamp': log.timestamp.isoformat(),
                    'operator': log.operator,
                    'hostname': log.hostname,
                    'operation_id': log.operation_id,
                    'target_id': log.target_id,
                    'phase': log.phase.value if log.phase else None,
                    'action_type': log.action_type.value,
                    'command': log.command,
                    'tool_name': log.tool_name,
                    'description': log.description,
                    'output': log.output,
                    'success': log.success,
                    'tags': log.tags,
                    'sensitive': log.sensitive
                }
                logs_data.append(log_dict)

            # Send to server
            response = requests.post(
                f'{server_url}/oplog/sync',
                json={'logs': logs_data},
                timeout=timeout
            )
            response.raise_for_status()

            result = response.json()

            # Mark logs as synced
            if result.get('success'):
                log_ids = [log.id for log in unsynced_logs if log.id]
                self.mark_synced(log_ids)

                logger.info(f"Successfully synced {result.get('synced_count', 0)} logs to {server_url}")

                return {
                    'success': True,
                    'synced_count': result.get('synced_count', 0),
                    'message': result.get('message', 'Sync successful')
                }
            else:
                return {
                    'success': False,
                    'synced_count': 0,
                    'message': 'Server reported sync failure',
                    'error': result.get('message', 'Unknown error')
                }

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to server: {server_url}")
            return {
                'success': False,
                'synced_count': 0,
                'message': 'Cannot connect to server',
                'error': 'Connection refused or server offline'
            }

        except requests.exceptions.Timeout:
            logger.error(f"Sync request timed out after {timeout}s")
            return {
                'success': False,
                'synced_count': 0,
                'message': 'Request timed out',
                'error': f'Server did not respond within {timeout} seconds'
            }

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during sync: {e}")
            return {
                'success': False,
                'synced_count': 0,
                'message': 'Server error',
                'error': f'HTTP {e.response.status_code}: {e.response.text}'
            }

        except Exception as e:
            logger.error(f"Unexpected error during sync: {e}")
            return {
                'success': False,
                'synced_count': 0,
                'message': 'Sync failed',
                'error': str(e)
            }
