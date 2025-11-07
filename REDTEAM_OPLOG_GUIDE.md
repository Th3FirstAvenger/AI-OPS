# Red Team OpLog - Guía de Uso

## 🎯 Descripción

AI-OPS ahora incluye un sistema completo de logging de operaciones para Red Team, permitiendo:

- **Terminal híbrido**: Ejecuta comandos directamente y usa el asistente de IA
- **Auto-logging**: Todos los comandos se loguean automáticamente
- **Logging manual**: Para acciones RDP, herramientas GUI, etc.
- **Contexto de operación**: Engagement, target, fase
- **Export para SOC**: JSON/CSV para correlación de alertas

## 🚀 Inicio Rápido

### 1. Configurar una Operación

```bash
# Iniciar AI-OPS CLI
python3 ai_ops_cli.py

# Crear una nueva operación
ai-ops > :op new
Operation name: ACME-PT-2025
Client: ACME Corp
Description: Penetration testing engagement

# Ver operaciones
ai-ops > :op list

# Establecer operación activa
[ACME-PT-2025] ai-ops > :op set
```

### 2. Configurar Target

```bash
# Crear un nuevo target
[ACME-PT-2025] ai-ops > :target new
Target name: DC01
IP address: 192.168.1.50
OS: Windows Server 2019

# Establecer target activo
[ACME-PT-2025] ai-ops > :target set
Target name: DC01

# Ver targets
[ACME-PT-2025] ai-ops > :target list
```

### 3. Establecer Fase de la Operación

```bash
# Establecer fase actual
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

### 4. Modo Shell (Auto-logging)

```bash
# Activar modo shell
[ACME-PT-2025] ai-ops > shell
Shell mode: enabled

# Todos los comandos se ejecutan y loguean automáticamente
[ACME-PT-2025] $ → DC01 > nmap -sV -p- 192.168.1.50
[ACME-PT-2025] $ → DC01 > crackmapexec smb 192.168.1.50 -u users.txt -p passwords.txt

# Salir del modo shell
[ACME-PT-2025] $ → DC01 > shell
Shell mode: disabled
```

### 5. Logging Manual

Para acciones que no son CLI (RDP, GUI tools, etc.):

```bash
# Log completo
[ACME-PT-2025] ai-ops > :log
Description: Acceso RDP a DC01 como administrator
Action type: rdp
Tool name (optional): mstsc
Tags (comma-separated, optional): lateral_movement, admin_access

# Nota rápida
[ACME-PT-2025] ai-ops > :note
Note: Encontrado hash NTLM del Domain Admin en memoria
```

### 6. Ver y Exportar Logs

```bash
# Ver logs recientes
[ACME-PT-2025] ai-ops > :logs
Number of logs to show: 50

# Ver estadísticas
[ACME-PT-2025] ai-ops > :stats

# Sincronizar logs al servidor central
[ACME-PT-2025] ai-ops > :sync
Found 47 unsynced log(s)
Sync to http://127.0.0.1:8000? (y/n): y
Syncing...
✓ Successfully synced 47 log(s)

# Exportar para el SOC (backup local)
[ACME-PT-2025] ai-ops > :export
Export format (json/csv): json
✓ Exported 127 logs to oplog_ACME-PT-2025_20251107.json
```

### 7. Usar el Asistente de IA

```bash
# Modo chat normal (mantiene la funcionalidad original)
[ACME-PT-2025] ai-ops > chat

# Preguntar al agente de IA
ACME-PT-2025 (1) > ¿Cómo puedo extraer credenciales de LSASS?
ACME-PT-2025 (1) > Genera un payload para bypass AMSI

# Volver al modo normal
ACME-PT-2025 (1) > back
```

## 📊 Tipos de Acciones Loguéables

- `command` - Comandos CLI (auto)
- `rdp` - Sesiones RDP
- `gui_tool` - Herramientas GUI (Burp, Metasploit GUI, etc.)
- `manual` - Acción manual
- `note` - Nota general
- `file_transfer` - Transferencia de archivos
- `exploit` - Ejecución de exploit
- `credential` - Obtención de credenciales
- `persistence` - Mecanismos de persistencia
- `lateral_movement` - Movimiento lateral
- `privesc` - Escalada de privilegios
- `exfil` - Exfiltración de datos

## 🎨 Características Avanzadas

### Filtrado de Auto-logging

Por defecto, estos comandos NO se loguean (son triviales):
- ls, cd, pwd, clear, exit, history

Puedes desactivar el auto-logging:
```bash
[ACME-PT-2025] ai-ops > :toggle autolog
Auto-logging: disabled
```

### Estructura de la Base de Datos

Los logs se guardan en: `~/.aiops/oplog/operations.db` (SQLite)

Cada entrada contiene:
- Timestamp
- Operador (usuario del sistema)
- Hostname (máquina del operador)
- Operation ID
- Target ID
- Fase
- Tipo de acción
- Comando/Descripción
- Output (primeros 500 chars)
- Tags
- Estado de sincronización

### Formato de Export

**JSON** (completo):
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

**CSV** (para import a Excel/SIEM):
```
timestamp,operator,hostname,action_type,command,description,target_id,phase
2025-11-07T14:30:00,john.doe,kali-ws01,command,nmap...,Executed: nmap...,3,exploitation
```

## 🌐 Sincronización con Servidor Central

### Configuración del Servidor

El servidor central debe ejecutar el backend de AI-OPS:

```bash
# En el servidor central
cd AI-OPS
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# O con Docker
docker-compose up -d
```

### Sincronización de Logs

**Manual (recomendado):**
```bash
# Sincronizar logs cuando termines tu sesión
[ACME-PT-2025] ai-ops > :sync
```

**Verificar estado de sync:**
```bash
# Ver cuántos logs faltan sincronizar
[ACME-PT-2025] ai-ops > :stats
Total logs: 127
Unsynced logs: 47  # <-- logs pendientes
```

### Endpoints de API Disponibles

El servidor central expone estos endpoints:

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/oplog/sync` | POST | Sincronizar logs desde cliente |
| `/oplog/logs` | GET | Obtener logs consolidados |
| `/oplog/operations` | GET | Listar todas las operaciones |
| `/oplog/stats` | GET | Estadísticas globales |
| `/oplog/targets` | GET | Listar todos los targets |
| `/oplog/health` | GET | Health check |

### Consultar Logs Consolidados (Red Team Lead)

Como Red Team Lead, puedes consultar todos los logs del equipo vía API:

```bash
# Ver todos los logs de una operación
curl http://servidor:8000/oplog/logs?operation_id=1&limit=100

# Ver logs de un operador específico
curl http://servidor:8000/oplog/logs?operator=john.doe

# Ver estadísticas globales
curl http://servidor:8000/oplog/stats

# Ver estadísticas de una operación
curl http://servidor:8000/oplog/stats?operation_id=1
```

### Arquitectura de Sincronización

```
┌─────────────────────────────────────────────┐
│  Operator 1 (Kali)                          │
│  - Local SQLite: ~/.aiops/oplog/operations.db
│  - Ejecuta comandos → auto-log             │
│  - :sync → envía al servidor               │
└──────────────┬──────────────────────────────┘
               │
               │ HTTP POST /oplog/sync
               ▼
┌─────────────────────────────────────────────┐
│  Servidor Central (Team Server)             │
│  - Base de datos centralizada               │
│  - Consolida logs de todos los operadores  │
│  - API REST para consultas                  │
└──────────────┬──────────────────────────────┘
               ▲
               │ HTTP POST /oplog/sync
               │
┌──────────────┴──────────────────────────────┐
│  Operator 2 (Windows)                       │
│  - Local SQLite: ~/.aiops/oplog/operations.db
│  - Ejecuta comandos → auto-log             │
│  - :sync → envía al servidor               │
└─────────────────────────────────────────────┘
```

### Offline-First Design

- **Funciona sin conexión**: Los logs se guardan localmente aunque el servidor esté caído
- **Sincronización diferida**: Cuando el servidor vuelve, ejecuta `:sync` para enviar todo
- **Sin pérdida de datos**: Todos los logs están en SQLite local como backup
- **Flag de sync**: Cada log tiene un flag `synced` para saber qué falta enviar

## 🔄 Workflow Recomendado

### Para el Red Team Lead:

1. **Inicio de Engagement**:
   - Crear operación con `:op new`
   - Compartir ID con el equipo
   - Crear targets iniciales con `:target new`

2. **Durante la Operación**:
   - Revisar actividad con `:logs` y `:stats`
   - Verificar que el equipo esté logueando

3. **Fin de Día/Engagement**:
   - Exportar logs con `:export`
   - Enviar al SOC para correlación
   - Archivar para el reporte final

### Para los Operadores:

1. **Inicio de Sesión**:
   ```bash
   :op set  # Establecer operación activa
   :target set  # Establecer target actual
   :phase set  # Establecer fase
   shell  # Activar modo shell
   ```

2. **Durante el Trabajo**:
   - Comandos automáticamente logueados
   - Para RDP/GUI: `:log` o `:note`
   - Cambiar target cuando sea necesario: `:target set`

3. **Fin de Sesión**:
   - Revisar logs del día: `:logs`
   - Asegurar que todo está documentado

## 🤖 Integración con IA

Puedes consultar al agente de IA sobre tus logs:

```bash
# Modo chat
chat

# Preguntas útiles
> Resume las acciones realizadas hoy contra DC01
> ¿Qué comandos fallaron en la fase de explotación?
> Genera un reporte ejecutivo de esta operación
> ¿Qué credenciales hemos obtenido hasta ahora?
```

## 🔒 Seguridad

- **Datos sensibles**: Usa el flag `sensitive` para marcar logs con credenciales
- **Ofuscación**: El sistema NO ofusca automáticamente (hazlo manualmente si es necesario)
- **Permisos**: La base de datos se crea con permisos 600 (solo tu usuario)
- **Offline-first**: Funciona sin conexión, sincroniza después

## 🚧 Próximas Funcionalidades

- [x] Servidor central para sincronización ✅
- [x] API REST para consulta de logs ✅
- [ ] Dashboard web para visualización
- [ ] Alertas automáticas (ej: credenciales obtenidas)
- [ ] Integración con SIEM (Splunk, ELK)
- [ ] Generación automática de reportes
- [ ] Auto-sync en background (opcional)

## 📝 Comandos Rápidos de Referencia

| Comando | Descripción |
|---------|-------------|
| `shell` | Toggle modo shell |
| `:op new/list/set/info` | Gestión de operaciones |
| `:target new/list/set` | Gestión de targets |
| `:phase set` | Establecer fase |
| `:log` | Log manual completo |
| `:note` | Nota rápida |
| `:logs` | Ver logs |
| `:stats` | Estadísticas |
| `:export` | Exportar para SOC |
| `:sync` | Sincronizar con servidor |
| `:toggle autolog` | Toggle auto-logging |
| `chat` | Asistente de IA |
| `help` | Ayuda completa |

## 💡 Tips

1. **Usa nombres descriptivos** para operations y targets
2. **Cambia la fase** según progresas para mejor organización
3. **Sincroniza regularmente** con `:sync` para consolidar logs del equipo
4. **Exporta como backup** para evitar pérdida de datos
5. **Agrega notas** para contexto que los comandos no capturan
6. **Revisa los logs** al final del día para verificar completitud
7. **Verifica el estado de sync** con `:stats` antes de terminar tu sesión

---

**¿Problemas o sugerencias?** Abre un issue en el repositorio.
