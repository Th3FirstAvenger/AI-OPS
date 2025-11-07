# Red Team OpLog - User Guide

## 🎯 Overview

AI-OPS now includes a comprehensive operation logging system for Red Team activities, featuring:

- **Hybrid Terminal**: Execute commands directly and use the AI assistant
- **Auto-logging**: All commands are automatically logged
- **Manual Logging**: For RDP sessions, GUI tools, etc.
- **Operation Context**: Engagement, target, and phase tracking
- **SOC Export**: JSON/CSV for alert correlation

## 🚀 Quick Start

### 1. Setup an Operation

```bash
# Start AI-OPS CLI
python3 ai_ops_cli.py

# Create a new operation
ai-ops > :op new
Operation name: ACME-PT-2025
Client: ACME Corp
Description: Penetration testing engagement

# List operations
ai-ops > :op list

# Set active operation
[ACME-PT-2025] ai-ops > :op set
```

### 2. Configure Target

```bash
# Create a new target
[ACME-PT-2025] ai-ops > :target new
Target name: DC01
IP address: 192.168.1.50
OS: Windows Server 2019

# Set active target
[ACME-PT-2025] ai-ops > :target set
Target name: DC01

# List targets
[ACME-PT-2025] ai-ops > :target list
```

### 3. Set Operation Phase

```bash
# Set current phase
[ACME-PT-2025] ai-ops > :phase set
Available phases:
  1. recon
  2. scanning
  3. exploitation
  4. post_exploitation
  5. persistence
  6. lateral_movement
  7. exfiltration
  8. cleanup
Select phase: 2
```

### 4. Shell Mode (Auto-logging)

```bash
# Enable shell mode
[ACME-PT-2025] ai-ops > shell
Shell mode: enabled

# All commands are executed and logged automatically
[ACME-PT-2025] $ → DC01 > nmap -sV -p- 192.168.1.50
[ACME-PT-2025] $ → DC01 > crackmapexec smb 192.168.1.50 -u users.txt -p passwords.txt

# Exit shell mode
[ACME-PT-2025] $ → DC01 > shell
Shell mode: disabled
```

### 5. Manual Logging

For non-CLI actions (RDP, GUI tools, etc.):

```bash
# Full log entry
[ACME-PT-2025] ai-ops > :log
Description: RDP access to DC01 as administrator
Action type: rdp
Tool name (optional): mstsc
Tags (comma-separated, optional): lateral_movement, admin_access

# Quick note
[ACME-PT-2025] ai-ops > :note
Note: Found NTLM hash for Domain Admin in memory
```

### 6. View and Export Logs

```bash
# View recent logs
[ACME-PT-2025] ai-ops > :logs
Number of logs to show: 50

# View statistics
[ACME-PT-2025] ai-ops > :stats

# Sync logs to central server
[ACME-PT-2025] ai-ops > :sync
Found 47 unsynced log(s)
Sync to http://127.0.0.1:8000? (y/n): y
Syncing...
✓ Successfully synced 47 log(s)

# Export for SOC (local backup)
[ACME-PT-2025] ai-ops > :export
Export format (json/csv): json
✓ Exported 127 logs to oplog_ACME-PT-2025_20251107.json
```

### 7. Use the AI Assistant

```bash
# Normal chat mode (keeps original functionality)
[ACME-PT-2025] ai-ops > chat

# Ask the AI agent
ACME-PT-2025 (1) > How can I extract credentials from LSASS?
ACME-PT-2025 (1) > Generate a payload for AMSI bypass

# Return to normal mode
ACME-PT-2025 (1) > back
```

## 📊 Loggable Action Types

- `command` - CLI commands (auto)
- `rdp` - RDP sessions
- `gui_tool` - GUI tools (Burp, Metasploit GUI, etc.)
- `manual` - Manual action
- `note` - General note
- `file_transfer` - File transfers
- `exploit` - Exploit execution
- `credential` - Credential harvesting
- `persistence` - Persistence mechanisms
- `lateral_movement` - Lateral movement
- `privesc` - Privilege escalation
- `exfil` - Data exfiltration

## 🎨 Advanced Features

### Auto-logging Filtering

By default, these commands are NOT logged (trivial):
- ls, cd, pwd, clear, exit, history

You can disable auto-logging:
```bash
[ACME-PT-2025] ai-ops > :toggle autolog
Auto-logging: disabled
```

### Database Structure

Logs are stored in: `~/.aiops/oplog/operations.db` (SQLite)

Each entry contains:
- Timestamp
- Operator (system username)
- Hostname (operator's machine)
- Operation ID
- Target ID
- Phase
- Action type
- Command/Description
- Output (first 500 chars)
- Tags
- Sync status

### Export Formats

**JSON** (complete):
```json
{
  "timestamp": "2025-11-07T14:30:00",
  "operator": "john.doe",
  "hostname": "kali-ws01",
  "operation_id": 1,
  "target_id": 3,
  "phase": "exploitation",
  "action_type": "command",
  "command": "nmap -sV -p- 192.168.1.50",
  "description": "Executed: nmap -sV -p- 192.168.1.50",
  "output": "...",
  "success": true,
  "tags": ["recon", "network"],
  "sensitive": false
}
```

**CSV** (for Excel/SIEM import):
```
timestamp,operator,hostname,action_type,command,description,target_id,phase
2025-11-07T14:30:00,john.doe,kali-ws01,command,nmap...,Executed: nmap...,3,exploitation
```

## 🌐 Central Server Synchronization

### Server Setup

The central server should run the AI-OPS backend:

```bash
# On the central server
cd AI-OPS
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up -d
```

### Log Synchronization

**Manual (recommended):**
```bash
# Sync logs when you finish your session
[ACME-PT-2025] ai-ops > :sync
```

**Check sync status:**
```bash
# See how many logs need syncing
[ACME-PT-2025] ai-ops > :stats
Total logs: 127
Unsynced logs: 47  # <-- pending logs
```

### Available API Endpoints

The central server exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/oplog/sync` | POST | Sync logs from client |
| `/oplog/logs` | GET | Get consolidated logs |
| `/oplog/operations` | GET | List all operations |
| `/oplog/stats` | GET | Global statistics |
| `/oplog/targets` | GET | List all targets |
| `/oplog/health` | GET | Health check |

### Query Consolidated Logs (Red Team Lead)

As Red Team Lead, you can query all team logs via API:

```bash
# View all logs for an operation
curl http://server:8000/oplog/logs?operation_id=1&limit=100

# View logs from a specific operator
curl http://server:8000/oplog/logs?operator=john.doe

# View global statistics
curl http://server:8000/oplog/stats

# View operation-specific statistics
curl http://server:8000/oplog/stats?operation_id=1
```

### Synchronization Architecture

```
┌─────────────────────────────────────────────┐
│  Operator 1 (Kali)                          │
│  - Local SQLite: ~/.aiops/oplog/operations.db
│  - Execute commands → auto-log              │
│  - :sync → send to server                   │
└──────────────┬──────────────────────────────┘
               │
               │ HTTP POST /oplog/sync
               ▼
┌─────────────────────────────────────────────┐
│  Central Server (Team Server)               │
│  - Centralized database                     │
│  - Consolidates logs from all operators     │
│  - REST API for queries                     │
└──────────────┬──────────────────────────────┘
               ▲
               │ HTTP POST /oplog/sync
               │
┌──────────────┴──────────────────────────────┐
│  Operator 2 (Windows)                       │
│  - Local SQLite: ~/.aiops/oplog/operations.db
│  - Execute commands → auto-log              │
│  - :sync → send to server                   │
└─────────────────────────────────────────────┘
```

### Offline-First Design

- **Works offline**: Logs are saved locally even if server is down
- **Deferred sync**: When server returns, run `:sync` to send everything
- **No data loss**: All logs are in local SQLite as backup
- **Sync flag**: Each log has a `synced` flag to track pending uploads

## 🔄 Recommended Workflow

### For Red Team Lead:

1. **Engagement Start**:
   - Create operation with `:op new`
   - Share ID with team
   - Create initial targets with `:target new`

2. **During Operation**:
   - Review activity with `:logs` and `:stats`
   - Verify team is logging properly

3. **End of Day/Engagement**:
   - Export logs with `:export`
   - Send to SOC for correlation
   - Archive for final report

### For Operators:

1. **Session Start**:
   ```bash
   :op set  # Set active operation
   :target set  # Set current target
   :phase set  # Set phase
   shell  # Enable shell mode
   ```

2. **During Work**:
   - Commands automatically logged
   - For RDP/GUI: `:log` or `:note`
   - Change target as needed: `:target set`

3. **Session End**:
   - Review daily logs: `:logs`
   - Ensure everything is documented
   - Sync to server: `:sync`

## 🤖 AI Integration

You can query the AI agent about your logs:

```bash
# Chat mode
chat

# Useful queries
> Summarize today's actions against DC01
> Which commands failed during exploitation phase?
> Generate executive report for this operation
> What credentials have we obtained so far?
```

## 🔒 Security

- **Sensitive data**: Use the `sensitive` flag to mark logs with credentials
- **Obfuscation**: System does NOT obfuscate automatically (do it manually if needed)
- **Permissions**: Database is created with 600 permissions (owner only)
- **Offline-first**: Works without connection, syncs later

## 🚧 Upcoming Features

- [x] Central server for synchronization ✅
- [x] REST API for log queries ✅
- [ ] Web dashboard for visualization
- [ ] Automatic alerts (e.g., credentials obtained)
- [ ] SIEM integration (Splunk, ELK)
- [ ] Automatic report generation
- [ ] Background auto-sync (optional)

## 📝 Quick Reference Commands

| Command | Description |
|---------|-------------|
| `shell` | Toggle shell mode |
| `:op new/list/set/info` | Operation management |
| `:target new/list/set` | Target management |
| `:phase set` | Set phase |
| `:log` | Full manual log |
| `:note` | Quick note |
| `:logs` | View logs |
| `:stats` | Statistics |
| `:export` | Export for SOC |
| `:sync` | Sync to server |
| `:toggle autolog` | Toggle auto-logging |
| `chat` | AI Assistant |
| `help` | Full help |

## 💡 Tips

1. **Use descriptive names** for operations and targets
2. **Change phase** as you progress for better organization
3. **Sync regularly** with `:sync` to consolidate team logs
4. **Export as backup** to avoid data loss
5. **Add notes** for context that commands don't capture
6. **Review logs** at end of day to verify completeness
7. **Check sync status** with `:stats` before ending your session

---

**Issues or suggestions?** Open an issue in the repository.
