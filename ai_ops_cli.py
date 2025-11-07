import argparse
import os
import sys
import json
import subprocess
from typing import Callable, Any, Dict, Type, Generator, Optional
from urllib.parse import urlparse

import requests
from requests.exceptions import ConnectionError, HTTPError
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import InvalidResponse, Prompt
from rich.markup import escape
from prompt_toolkit.formatted_text import ANSI
from rich.tree import Tree
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers import MarkdownLexer
from prompt_toolkit.keys import Keys
from pydantic import BaseModel, validate_call

# Import OpLog functionality
from src.core.oplog import OperationLog, LogEntry, Operation, Target
from src.core.oplog.models import ActionType, Phase, OperationContext

VERSION = "0.1.0-redteam"


def build_input_multiline(current_session, api_url, model_name):
    """Creates an input prompt that incluye session name, API URL and model name"""
    bindings = KeyBindings()

    @bindings.add(Keys.ControlDown, eager=True)
    def _(event):
        event.current_buffer.newline()

    return PromptSession(
        ANSI(
            f"\033[38;5;124m {current_session.get('name', 'unknown')}\033[38;5;245m@\033[38;5;246m{api_url} :: "
            f"\033[38;5;160m[{model_name}]\n"
            f"\033[38;5;196m> \033[0m"
        ),
        key_bindings=bindings,
        lexer=PygmentsLexer(MarkdownLexer)
    )




class CommandCompleter(Completer):
    def __init__(self, commands):
        self.commands = commands

    def get_completions(self, document, complete_event):
        # Get word being completed
        word = document.get_word_before_cursor()
        
        # Return matching commands
        for cmd in self.commands:
            if cmd.startswith(word):
                yield Completion(cmd, start_position=-len(word))


class AgentClient:
    """Client for Agent API with Red Team Operation Logging"""

    def __init__(self, api_url: str = 'http://127.0.0.1:8000', model_name: str = 'mistral'):
        self.api_url = api_url
        self.client = requests.Session()
        self.console = Console(force_terminal=True)
        self.current_session = {'sid': 0, 'name': 'Undefined'}
        self.model_name = model_name

        # Initialize OpLog
        self.oplog = OperationLog()
        self.opcontext = OperationContext()

        # Load active operation if exists
        active_op = self.oplog.get_active_operation()
        if active_op:
            self.opcontext.operation = active_op

        self.multiline_input = build_input_multiline(
            self.current_session,
            self.api_url,
            self.model_name
        )

        self.show_thinking = "deepseek" in self.model_name.lower()

        # Shell mode flag
        self.shell_mode = False

        self.commands = {
            'help': self.help,
            'clear': AgentClient.clear_terminal,
            'exit': '',

            'chat': self.chat,

            'new': self.new_session,
            'save': self.save_session,
            'delete': self.delete_session,
            'rename': self.rename_session,
            'list sessions': self.list_sessions,
            'load': self.load_session,

            # Shell mode
            'shell': self.toggle_shell_mode,

            # Red Team OpLog commands
            ':op new': self.op_new,
            ':op list': self.op_list,
            ':op set': self.op_set,
            ':op info': self.op_info,
            ':target set': self.target_set,
            ':target new': self.target_new,
            ':target list': self.target_list,
            ':phase set': self.phase_set,
            ':log': self.manual_log,
            ':note': self.add_note,
            ':logs': self.view_logs,
            ':stats': self.view_stats,
            ':export': self.export_logs,
            ':toggle autolog': self.toggle_autolog,

            # RAG is disabled in the current version
            # 'list collections': self.__list_collections,
            # 'create collection': self.__create_collection

            'toggle thinking': self.toggle_thinking,
        }

        self.console.print("[bold blue]ai-ops-cli[/] (beta) starting.")
        # Display thinking status on startup
        self.console.print(f"Thinking mode: [{'green' if self.show_thinking else 'red'}]{'On' if self.show_thinking else 'Off'}[/]")
        try:
            response = self.client.get(f'{self.api_url}/ping', timeout=5)
            response.raise_for_status()
            self.console.print(f"Backend: [blue]online[/]")
            self.console.print(
                "[bold cyan]ℹ️  Tip:[/bold cyan] Press [bold green]Ctrl + ↓ (Down Arrow)[/bold green] to move to the next line while typing.",
                style="italic"
            )
        except (ConnectionError, HTTPError):
            self.console.print('Backend: [red]offline[/]')
            sys.exit(-1)
        self.console.print()
   
    def run(self):
        """Runs the main loop of the client"""
        # Create a history object for command history
        history = InMemoryHistory()
        
        # Create command completer for tab completion
        completer = CommandCompleter(self.commands.keys())
        
        # Set in_chat flag
        self.in_chat = False
        
        while True:
            try:
                # Use prompt_toolkit with dynamic prompt
                user_input = PromptSession(
                    history=history,
                    completer=completer,
                    complete_while_typing=True
                ).prompt(self.prompt_text())
            
                # Remove any control characters
                user_input = ''.join(ch for ch in user_input if ord(ch) >= 32)
                
                if not user_input:
                    continue
                    
                if user_input == 'exit':
                    break

                # Shell mode: execute commands directly
                if self.shell_mode and not user_input.startswith(':') and user_input not in ['shell', 'help', 'clear', 'exit']:
                    self.execute_shell_command(user_input)
                # Execute command if valid
                elif user_input in self.commands:
                    self.commands[user_input]()
                else:
                    closest = self.find_closest_command(user_input)
                    if closest:
                        self.console.print(f"Using command: [bold blue]{closest}[/]")
                        self.commands[closest]()
                    else:
                        # In shell mode, try to execute as shell command
                        if self.shell_mode:
                            self.execute_shell_command(user_input)
                        else:
                            self.console.print('Command not recognized. Try "help" for available commands.', style='bold red')
                            self.commands['help']()
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f'[red]Error: {e}[/]')
                continue

    def find_closest_command(self, input_text):
        """Find the closest matching command for a given input"""
        # Exact match first
        if input_text in self.commands:
            return input_text
            
        # Check for partial matches
        partial_matches = [cmd for cmd in self.commands.keys() if input_text in cmd]
        if len(partial_matches) == 1:
            return partial_matches[0]
            
        # Look for close matches (allowing for typos)
        close_matches = []
        for cmd in self.commands.keys():
            # Calculate how different the input is from each command
            if len(input_text) > 2 and len(cmd) > 2:  # Only for inputs of reasonable length
                # Simple distance calculation - count different characters
                distance = sum(1 for a, b in zip(input_text, cmd) if a != b)
                # Add penalty for length difference
                distance += abs(len(input_text) - len(cmd))
                
                if distance <= 3:  # Allow up to 3 differences
                    close_matches.append((cmd, distance))
        
        # Sort by distance (closest first)
        close_matches.sort(key=lambda x: x[1])
        
        # Return the closest match if there is one
        if close_matches:
            return close_matches[0][0]
            
        return None
    

    def new_session(self):
        """Creates a new session and updates the prompt"""
        session_name = Prompt.ask(
            'Session Name',
            console=self.console
        )

        response = self.client.post(
            f'{self.api_url}/sessions',
            params={'name': session_name}
        )
        response.raise_for_status()

       
        self.current_session = {'sid': response.json()['sid'], 'name': session_name}

        
        self.multiline_input = build_input_multiline(
            self.current_session, 
            self.api_url, 
            self.model_name  
        )

        self.chat(print_name=True) 


    def save_session(self):
        """Save the current session"""
        response = self.client.put(
            f'{self.api_url}/sessions/{self.current_session["sid"]}/chat'
        )
        if response.status_code != 200:
            self.console.print(f'[!] Failed: {response.status_code}')
        else:
            self.console.print(f'[+] Saved')

    def rename_session(self):
        """Renames the current session"""
        session_name = Prompt.ask(
            'New Name',
            console=self.console
        )

        response = self.client.put(
            f'{self.api_url}/sessions/{self.current_session["sid"]}',
            params={'new_name': str(session_name)}
        )
        response.raise_for_status()
        self.current_session['name'] = str(session_name)
        self.chat(print_name=True)

    def delete_session(self):
        """Deletes a session"""
        session_id = Prompt.ask(
            'Enter session ID',
            console=self.console
        )
        if not session_id.isdigit():
            self.console.print('[-] Not a number', style='bold red')

        response = self.client.delete(
            f'{self.api_url}/sessions/{session_id}'
        )
        response.raise_for_status()
        body = response.json()
        self.console.print(f'[{"+" if body["success"] else "-"}] {body["message"]}')

    def list_sessions(self):
        """List all sessions"""
        response = self.client.get(
            f'{self.api_url}/sessions'
        )
        response.raise_for_status()
        body = response.json()
        if len(body) == 0:
            self.console.print('[+] No sessions found')
        else:
            tree = Tree("[+] Available Sessions:")
            for session in body:
                tree.add(f'({session["sid"]}) {session["name"]}')
            self.console.print(tree)

    def load_session(self):
        """Opens an existing session"""
        session_id = Prompt.ask(
            'Enter session ID',
            console=self.console
        )
        if not session_id.isdigit():
            self.console.print('[-] Not a number', style='bold red')
            self.load_session()

        response = self.client.get(
            f'{self.api_url}/sessions/{int(session_id)}/chat',
        )
        response.raise_for_status()

        body = response.json()
        if 'success' in body:
            self.console.print(f'No session for {session_id}', style='red')

        sid = body['sid']
        name = body['name']
        self.current_session = {'sid': sid, 'name': name}
        self.console.print(f'({sid}) [bold blue]{name}[/]')

        body['messages'] = body['messages'][1:]  # exclude system message
        for msg in body['messages']:
            self.console.print(f'[bold white]{msg["role"]}[/]: {msg["content"]}\n')
        self.chat(print_name=False)

        
    def chat(self, print_name=True):
        """Opens a chat with the Agent"""
        sid = self.current_session["sid"]
        query_url = f'{self.api_url}/sessions/{sid}/chat'

        if print_name:
            name = self.current_session["name"]
            self.console.print(f'Session: [bold blue]{name}[/] (ID: {sid})')

        # Set in chat mode
        self.in_chat = True
        
        try:
            while True:
                q = self.multiline_input.prompt()
                if q.startswith('back'):
                    break
                self.__generate_response(query_url, q)
        finally:
            # Always reset chat mode when exiting
            self.in_chat = False

    def toggle_thinking(self):
        """Toggle the display of LLM thinking process"""
        self.show_thinking = not self.show_thinking
        status = "On" if self.show_thinking else "Off"
        status_color = "green" if self.show_thinking else "red"
        self.console.print(f"Thinking mode: [{status_color}]{status}[/]")

    def __generate_response(self, url: str, query: str):
        """Generate a response from the API with thinking process handling"""
        try:
            with self.client.post(
                    url,
                    json={'query': query},
                    headers=None,
                    stream=True
            ) as resp:
                resp.raise_for_status()

                response_text = '**Assistant**: '
                
                # Variables to track thinking blocks
                in_thinking = False
                thinking_buffer = ""
                
                with Live(console=self.console, refresh_per_second=10) as live:
                    live.update(Markdown(response_text))
                    for chunk in resp.iter_content(decode_unicode=True):
                        if chunk:
                            try:
                                # Convert bytes to string if needed
                                if isinstance(chunk, bytes):
                                    chunk = chunk.decode('utf-8')
                                
                                # Process the chunk character by character to handle thinking tags
                                for char in chunk:
                                    if not in_thinking:
                                        # Look for start of thinking tag
                                        if response_text.endswith("<think"):
                                            response_text += char
                                            if response_text.endswith("<think>"):
                                                in_thinking = True
                                                thinking_buffer = ""
                                                # Remove the tag from visible response
                                                response_text = response_text[:-7]
                                        else:
                                            response_text += char
                                    else:  # We're inside a thinking block
                                        thinking_buffer += char
                                        # Check for end of thinking tag
                                        if thinking_buffer.endswith("</think>"):
                                            in_thinking = False
                                            # Extract the thinking content without the end tag
                                            thinking_content = thinking_buffer[:-8]
                                            
                                            # Display thinking if enabled
                                            if self.show_thinking:
                                                self.console.print("\n[bold yellow]Thinking:[/bold yellow]", style="yellow")
                                                self.console.print(thinking_content, style="dim yellow")
                                                self.console.print("[yellow]End of thinking[/yellow]\n")
                                            
                                            thinking_buffer = ""
                                            # Update the live display without the thinking content
                                            live.update(Markdown(response_text))
                                
                                # Update the live display with the processed content
                                live.update(Markdown(response_text))
                            except UnicodeDecodeError:
                                pass

                print()
        except requests.exceptions.HTTPError:
            if 400 <= resp.status_code < 500:
                self.console.print(f'[red]Client Error: {resp.status_code}[/]')
            else:
                self.console.print(f'[red]Server Error: {resp.status_code}[/]')
        except requests.exceptions.ConnectionError:
            self.console.print('[red]Connection Error[/]')
        except requests.exceptions.Timeout:
            self.console.print('[red]Timeout Error[/]')
        except requests.exceptions.RequestException as e:
            self.console.print(f'[red]Error: {e}[/]')
        except KeyboardInterrupt:
            self.console.print('[red]Interrupted[/]')
        except Exception as e:
            self.console.print(f'[red]Error: {e}[/]')
        
    # RAG is disabled in the current version
    def __list_collections(self):
        """Know what collections are available"""
        response = self.client.get(
            f'{self.api_url}/collections/list/'
        )
        response.raise_for_status()
        body = response.json()
        if len(body) == 0:
            self.console.print('[+] No collections found')
        else:
            tree = Tree("[+] Available Collections:")
            for collection in body:
                c_doc = f"[bold blue]{collection['title']}[/]\n"

                str_topics = ', '.join(collection['topics'])
                c_doc += f"[bold white]Topics[/]: {str_topics}\n"

                c_doc += "[bold white]Documents[/]:\n"
                for document in collection['documents']:
                    c_doc += f"- {document['name']}\n"

                tree.add(c_doc)

            self.console.print(tree)

    def __create_collection(self):
        """Upload a collection to RAG"""
        collection_title = Prompt.ask(
            prompt='Title: ',
            console=self.console
        )
        collection_path = Prompt.ask(
            prompt='Path (leave blank for nothing): ',
            console=self.console
        )

        try:
            if collection_path:
                with open(collection_path, 'rb') as collection_file:
                    response = requests.post(
                        url=f'{self.api_url}/collections/new',
                        data={'title': collection_title},
                        files={'file': collection_file}
                    )
            else:
                response = requests.post(
                    url=f'{self.api_url}/collections/new',
                    data={'title': collection_title}
                )

            response.raise_for_status()
            body: dict = response.json()

            if 'error' in body:
                self.console.print(f"[bold red][!] Failed: [/] {body['error']}")
            else:
                self.console.print(f"[bold blue][+] Success: [/] {body['success']}")

        except OSError as err:
            self.console.print(f"[bold red][!] Failed: [/] {err}")
        except requests.exceptions.HTTPError as http_err:
            self.console.print(f"[bold red][!] HTTP Error: [/] {http_err}")
        except requests.exceptions.RequestException as req_err:
            self.console.print(f"[bold red][!] Request Error: [/] {req_err}")

    def prompt_text(self):
        """Returns a stylized prompt text with operation context"""
        if hasattr(self, 'in_chat') and self.in_chat:
            session_name = self.current_session.get('name', 'Unknown')
            session_id = self.current_session.get('sid', '?')
            return ANSI(f"\033[38;5;196m\033[1m {session_name} \033[0m (\033[90m{session_id}\033[0m) > ")
        elif self.shell_mode:
            # Shell mode with operation context
            prefix = "\033[38;5;196m\033[1m $\033[0m"
            if self.opcontext.operation:
                op_name = self.opcontext.operation.name[:15]
                prefix = f"\033[38;5;208m[{op_name}]\033[0m {prefix}"
            if self.opcontext.current_target:
                target_name = self.opcontext.current_target.name
                prefix = f"{prefix} \033[38;5;33m→ {target_name}\033[0m"
            return ANSI(f"{prefix} > ")
        else:
            # Normal mode
            prefix = "\033[38;5;196m\033[1m ai-ops \033[0m"
            if self.opcontext.operation:
                op_name = self.opcontext.operation.name[:15]
                prefix = f"\033[38;5;208m[{op_name}]\033[0m {prefix}"
            return ANSI(f"{prefix} > ")

    # ==================== SHELL MODE ====================

    def toggle_shell_mode(self):
        """Toggle shell mode for direct command execution"""
        self.shell_mode = not self.shell_mode
        status = "enabled" if self.shell_mode else "disabled"
        color = "green" if self.shell_mode else "red"
        self.console.print(f"Shell mode: [{color}]{status}[/]")
        if self.shell_mode:
            self.console.print("[dim]Commands will be executed directly. Use 'shell' again to exit.[/]")

    def execute_shell_command(self, command: str):
        """Execute a shell command and log it"""
        try:
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = result.stdout if result.stdout else result.stderr
            success = result.returncode == 0

            # Display output
            if output:
                self.console.print(output)

            # Auto-log if enabled
            if self.opcontext.auto_log:
                # Check if command should be filtered
                cmd_base = command.split()[0] if command.split() else command
                if cmd_base not in self.opcontext.log_filter:
                    self.oplog.add_log(
                        action_type=ActionType.COMMAND,
                        description=f"Executed: {command}",
                        command=command,
                        output=output[:500] if output else None,  # Limit output size
                        success=success,
                        operation_id=self.opcontext.operation.id if self.opcontext.operation else None,
                        target_id=self.opcontext.current_target.id if self.opcontext.current_target else None,
                        phase=self.opcontext.current_phase
                    )

            return success

        except subprocess.TimeoutExpired:
            self.console.print("[red]Command timed out (30s limit)[/]")
            return False
        except Exception as e:
            self.console.print(f"[red]Error executing command: {e}[/]")
            return False

    # ==================== OPERATION MANAGEMENT ====================

    def op_new(self):
        """Create a new operation/engagement"""
        name = Prompt.ask("Operation name", console=self.console)
        client = Prompt.ask("Client (optional)", console=self.console, default="")
        description = Prompt.ask("Description (optional)", console=self.console, default="")

        operation = Operation(
            name=name,
            client=client if client else None,
            description=description if description else None,
            is_active=True
        )

        op_id = self.oplog.create_operation(operation)
        operation.id = op_id
        self.opcontext.operation = operation

        self.console.print(f"[green]✓[/] Created operation: [bold]{name}[/] (ID: {op_id})")

    def op_list(self):
        """List all operations"""
        operations = self.oplog.list_operations()

        if not operations:
            self.console.print("[yellow]No operations found[/]")
            return

        table = Table(title="Operations")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Client", style="dim")
        table.add_column("Start Date", style="green")
        table.add_column("Active", style="yellow")

        for op in operations:
            table.add_row(
                str(op.id),
                op.name,
                op.client or "-",
                op.start_date.strftime("%Y-%m-%d"),
                "✓" if op.is_active else ""
            )

        self.console.print(table)

    def op_set(self):
        """Set active operation"""
        op_id = Prompt.ask("Operation ID", console=self.console)

        if not op_id.isdigit():
            self.console.print("[red]Invalid ID[/]")
            return

        operation = self.oplog.get_operation(int(op_id))
        if not operation:
            self.console.print("[red]Operation not found[/]")
            return

        self.oplog.set_active_operation(int(op_id))
        self.opcontext.operation = operation

        self.console.print(f"[green]✓[/] Active operation: [bold]{operation.name}[/]")

    def op_info(self):
        """Show current operation info"""
        if not self.opcontext.operation:
            self.console.print("[yellow]No active operation[/]")
            return

        op = self.opcontext.operation
        stats = self.oplog.get_stats(operation_id=op.id)

        self.console.print(f"\n[bold]Operation:[/] {op.name}")
        self.console.print(f"[bold]Client:[/] {op.client or 'N/A'}")
        self.console.print(f"[bold]Start Date:[/] {op.start_date.strftime('%Y-%m-%d %H:%M')}")
        self.console.print(f"[bold]Description:[/] {op.description or 'N/A'}")
        self.console.print(f"\n[bold]Statistics:[/]")
        self.console.print(f"  Total logs: {stats['total']}")
        self.console.print(f"  Unsynced: {stats['unsynced']}")
        if stats['by_type']:
            self.console.print(f"  By type: {stats['by_type']}")
        self.console.print()

    # ==================== TARGET MANAGEMENT ====================

    def target_new(self):
        """Create a new target"""
        name = Prompt.ask("Target name/hostname", console=self.console)
        ip = Prompt.ask("IP address (optional)", console=self.console, default="")
        os_type = Prompt.ask("OS (optional)", console=self.console, default="")
        notes = Prompt.ask("Notes (optional)", console=self.console, default="")

        target = Target(
            name=name,
            ip_address=ip if ip else None,
            os=os_type if os_type else None,
            notes=notes if notes else None
        )

        target_id = self.oplog.create_target(target)
        target.id = target_id

        self.console.print(f"[green]✓[/] Created target: [bold]{name}[/] (ID: {target_id})")

    def target_set(self):
        """Set current target"""
        name = Prompt.ask("Target name", console=self.console)

        target = self.oplog.get_target_by_name(name)
        if not target:
            create = Prompt.ask(
                f"Target '{name}' not found. Create it?",
                choices=["y", "n"],
                default="y",
                console=self.console
            )
            if create == "y":
                ip = Prompt.ask("IP address (optional)", console=self.console, default="")
                target = Target(name=name, ip_address=ip if ip else None)
                target_id = self.oplog.create_target(target)
                target.id = target_id
            else:
                return

        self.opcontext.current_target = target
        self.console.print(f"[green]✓[/] Current target: [bold]{target.name}[/]")

    def target_list(self):
        """List all targets"""
        targets = self.oplog.list_targets()

        if not targets:
            self.console.print("[yellow]No targets found[/]")
            return

        table = Table(title="Targets")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("IP", style="green")
        table.add_column("OS", style="yellow")

        for target in targets:
            table.add_row(
                str(target.id),
                target.name,
                target.ip_address or "-",
                target.os or "-"
            )

        self.console.print(table)

    # ==================== PHASE MANAGEMENT ====================

    def phase_set(self):
        """Set current operation phase"""
        phases = [p.value for p in Phase]

        self.console.print("[bold]Available phases:[/]")
        for i, phase in enumerate(phases, 1):
            self.console.print(f"  {i}. {phase}")

        choice = Prompt.ask("Select phase", console=self.console)

        if choice.isdigit() and 1 <= int(choice) <= len(phases):
            phase = Phase(phases[int(choice) - 1])
        elif choice in phases:
            phase = Phase(choice)
        else:
            self.console.print("[red]Invalid phase[/]")
            return

        self.opcontext.current_phase = phase
        self.console.print(f"[green]✓[/] Current phase: [bold]{phase.value}[/]")

    # ==================== LOGGING ====================

    def manual_log(self):
        """Add a manual log entry"""
        description = Prompt.ask("Description", console=self.console)

        # Action type
        action_types = [t.value for t in ActionType]
        self.console.print("\n[bold]Action types:[/] " + ", ".join(action_types))
        action_type_str = Prompt.ask("Action type", console=self.console, default="manual")

        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.MANUAL

        # Optional fields
        tool = Prompt.ask("Tool name (optional)", console=self.console, default="")
        tags_str = Prompt.ask("Tags (comma-separated, optional)", console=self.console, default="")
        tags = [t.strip() for t in tags_str.split(",")] if tags_str else []

        self.oplog.add_log(
            action_type=action_type,
            description=description,
            tool_name=tool if tool else None,
            operation_id=self.opcontext.operation.id if self.opcontext.operation else None,
            target_id=self.opcontext.current_target.id if self.opcontext.current_target else None,
            phase=self.opcontext.current_phase,
            tags=tags
        )

        self.console.print("[green]✓[/] Log entry added")

    def add_note(self):
        """Add a quick note"""
        note = Prompt.ask("Note", console=self.console)

        self.oplog.add_log(
            action_type=ActionType.NOTE,
            description=note,
            operation_id=self.opcontext.operation.id if self.opcontext.operation else None,
            target_id=self.opcontext.current_target.id if self.opcontext.current_target else None,
            phase=self.opcontext.current_phase
        )

        self.console.print("[green]✓[/] Note added")

    def view_logs(self):
        """View recent logs"""
        limit_str = Prompt.ask("Number of logs to show", console=self.console, default="20")
        limit = int(limit_str) if limit_str.isdigit() else 20

        logs = self.oplog.get_logs(
            operation_id=self.opcontext.operation.id if self.opcontext.operation else None,
            limit=limit
        )

        if not logs:
            self.console.print("[yellow]No logs found[/]")
            return

        table = Table(title=f"Recent Logs ({len(logs)})")
        table.add_column("Time", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Target", style="yellow")

        for log in logs:
            target_name = "-"
            if log.target_id:
                # We'd need to fetch target name, for now just show ID
                target_name = f"T{log.target_id}"

            table.add_row(
                log.timestamp.strftime("%H:%M:%S"),
                log.action_type.value,
                log.description[:60] + "..." if len(log.description) > 60 else log.description,
                target_name
            )

        self.console.print(table)

    def view_stats(self):
        """View operation statistics"""
        stats = self.oplog.get_stats(
            operation_id=self.opcontext.operation.id if self.opcontext.operation else None
        )

        self.console.print("\n[bold]Operation Statistics[/]")
        self.console.print(f"Total logs: {stats['total']}")
        self.console.print(f"Unsynced logs: {stats['unsynced']}")

        if stats['by_type']:
            self.console.print("\n[bold]By Action Type:[/]")
            for action_type, count in stats['by_type'].items():
                self.console.print(f"  {action_type}: {count}")

        self.console.print()

    def export_logs(self):
        """Export logs for SOC"""
        format_type = Prompt.ask(
            "Export format",
            choices=["json", "csv"],
            default="json",
            console=self.console
        )

        logs = self.oplog.get_logs(
            operation_id=self.opcontext.operation.id if self.opcontext.operation else None,
            limit=10000
        )

        if not logs:
            self.console.print("[yellow]No logs to export[/]")
            return

        filename = f"oplog_{self.opcontext.operation.name if self.opcontext.operation else 'all'}_{logs[0].timestamp.strftime('%Y%m%d')}.{format_type}"

        if format_type == "json":
            import json
            with open(filename, 'w') as f:
                json.dump([log.model_dump() for log in logs], f, indent=2, default=str)
        else:
            # CSV export
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'operator', 'hostname', 'action_type',
                    'command', 'description', 'target_id', 'phase'
                ])
                writer.writeheader()
                for log in logs:
                    writer.writerow({
                        'timestamp': log.timestamp.isoformat(),
                        'operator': log.operator,
                        'hostname': log.hostname,
                        'action_type': log.action_type.value,
                        'command': log.command or '',
                        'description': log.description,
                        'target_id': log.target_id or '',
                        'phase': log.phase.value if log.phase else ''
                    })

        self.console.print(f"[green]✓[/] Exported {len(logs)} logs to [bold]{filename}[/]")

    def toggle_autolog(self):
        """Toggle automatic command logging"""
        self.opcontext.auto_log = not self.opcontext.auto_log
        status = "enabled" if self.opcontext.auto_log else "disabled"
        color = "green" if self.opcontext.auto_log else "red"
        self.console.print(f"Auto-logging: [{color}]{status}[/]")

    def help(self):
        """Print help message"""
        # Basic Commands
        self.console.print("\n[bold white]Basic Commands[/]")
        self.console.print("- [bold blue]help[/]   : Show available commands.")
        self.console.print("- [bold blue]clear[/]  : Clears the terminal.")
        self.console.print("- [bold blue]exit[/]   : Exit the program")
        self.console.print("- [bold blue]shell[/]  : Toggle shell mode (execute commands directly)")

        # Agent Related
        self.console.print("\n[bold white]AI Agent[/]")
        self.console.print("- [bold blue]chat[/]            : Open chat with the AI agent.")
        self.console.print("- [bold blue]back[/]            : Exit chat")
        self.console.print("- [bold blue]toggle thinking[/] : Toggle thinking mode")

        # Session Related
        self.console.print("\n[bold white]AI Sessions[/]")
        self.console.print("- [bold blue]new[/]             : Create a new AI session.")
        self.console.print("- [bold blue]save[/]            : Save the current session.")
        self.console.print("- [bold blue]load[/]            : Opens a session.")
        self.console.print("- [bold blue]delete[/]          : Delete the current session.")
        self.console.print("- [bold blue]rename[/]          : Rename the current session.")
        self.console.print("- [bold blue]list sessions[/]   : Show the saved sessions.")

        # Red Team OpLog
        self.console.print("\n[bold red]Red Team Operations[/]")
        self.console.print("- [bold cyan]:op new[/]         : Create a new operation/engagement")
        self.console.print("- [bold cyan]:op list[/]        : List all operations")
        self.console.print("- [bold cyan]:op set[/]         : Set active operation")
        self.console.print("- [bold cyan]:op info[/]        : Show current operation info")

        self.console.print("\n[bold red]Target Management[/]")
        self.console.print("- [bold cyan]:target new[/]     : Create a new target")
        self.console.print("- [bold cyan]:target set[/]     : Set current target")
        self.console.print("- [bold cyan]:target list[/]    : List all targets")

        self.console.print("\n[bold red]Operation Logging[/]")
        self.console.print("- [bold cyan]:phase set[/]      : Set operation phase (recon, exploitation, etc.)")
        self.console.print("- [bold cyan]:log[/]            : Add manual log entry (RDP, GUI tools, etc.)")
        self.console.print("- [bold cyan]:note[/]           : Add a quick note")
        self.console.print("- [bold cyan]:logs[/]           : View recent logs")
        self.console.print("- [bold cyan]:stats[/]          : View operation statistics")
        self.console.print("- [bold cyan]:export[/]         : Export logs for SOC (JSON/CSV)")
        self.console.print("- [bold cyan]:toggle autolog[/] : Toggle automatic command logging")

        self.console.print("\n[dim]In shell mode, commands are executed directly and auto-logged.[/]")
        self.console.print("\n")

    @staticmethod
    def clear_terminal():
        os.system(
            'cls' if os.name == 'nt'    # windows (its always him)
            else 'clear'                # unix
        )


class ValidateURLAction(argparse.Action):
    """
    Checks if the URL string for the API is valid.
    A valid url in this context has:
    - http/https scheme
    - the path is empty
    """

    def __call__(self, parser, namespace, values, option_string=None):
        parsed = urlparse(values)
        url_scheme = parsed.scheme
        url_path = parsed.path

        try:
            valid_scheme = url_scheme in ('http', 'https') if url_scheme else False
            valid_path = len(url_path) <= 1  # consider "/"
            assert valid_scheme and valid_path
        except AssertionError:
            print(f'[!] Invalid URL: {values}')
            sys.exit(-1)

        setattr(namespace, self.dest, values)


def main():
    """Main function for AI-OPS CLI client"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--api',
        default='http://127.0.0.1:8000',
        help='The Agent API address',
        action=ValidateURLAction
    )

    try:
        args = parser.parse_args(sys.argv[1:])
        model = os.getenv('MODEL', 'mistral')  # Usa os.getenv en vez de acceder a os.environ directamente

        client = AgentClient(api_url=args.api, model_name=model)
        client.run()
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()
