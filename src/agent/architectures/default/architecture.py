import re
import json
import logging
from typing import Generator, Dict, Tuple, List

from tool_parse import ToolRegistry

from src.agent import AgentArchitecture
from src.core import Conversation
from src.core.llm import LLM
from src.core.memory import Message, Role
from src.utils import get_logger, LOGS_PATH

logger = get_logger(__name__)


class State:
    """
    A class to manage the state of the LLM response stream.

    Attributes:
        SPEAKING (int): The assistant is currently generating a response.
        THINKING (int): The assistant is processing thoughts.
        IDLE (int): Indicates a transition between SPEAKING-THINKING.
    """
    SPEAKING = 1
    THINKING = 0
    IDLE = 2

    def __init__(self):
        self.__state = State.SPEAKING
        self.__count = 0

    def state(self, c: str):
        """:returns the state of the generation stream"""
        match self.__state:
            case State.SPEAKING:
                if c == '@':
                    # case it reached three @
                    if self.__count == 2:
                        self.__count = 0
                        self.__state = State.THINKING
                        return State.THINKING
                    # case it is parsing @
                    self.__count += 1
                    return State.IDLE
                else:
                    self.__count = 0
                    return State.SPEAKING
            case State.THINKING:
                if c == '@':
                    # case it reached three @
                    if self.__count == 2:
                        self.__count = 0
                        self.__state = State.SPEAKING
                        return State.IDLE
                    # case it is parsing #
                    self.__count += 1
                    return State.IDLE
                else:
                    self.__count = 0
                    return State.THINKING


class DefaultArchitecture(AgentArchitecture):
    """
    TODO: fix architecture since its kinda broken
        - search adds two user messages ???

    The overall code sucks. My bad.
    """
    model: str
    architecture_name = 'default_architecture'

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        router_prompt: str,
        general_prompt: str,
        reasoning_prompt: str,
        tool_prompt: str
    ):
        super().__init__()
        self.llm: LLM = llm
        self.model = llm.model
        self.__tool_registry: ToolRegistry = tools
        self.__tools: tuple = tuple(self.__tool_registry.marshal('base'))
        self.__prompts: Dict[str, str] = {
            'router': router_prompt,
            'general': general_prompt,
            'reasoning': reasoning_prompt,
            'tool': tool_prompt
        }

        self.__thought_parser: State = State()
        self.__tool_pattern = r'(?:{"name"|name).*"parameters".*}'

        tool_names = ', '.join([
            tool["function"]["name"]
            for tool in self.__tools
        ])
        logger.info(
            f'Initialized DefaultArchitecture with model {llm.model} and tools {tool_names}'
        )

        self.token_logger = logging.Logger('token_logger')
        formatter = logging.Formatter(f'%(name)s - {self.model}: %(message)s')

        logger_handler = logging.FileHandler(
            filename=f'{str(LOGS_PATH)}/token_usage.log',
            mode='a',
            encoding='utf-8'
        )
        logger_handler.setLevel(logging.DEBUG)
        logger_handler.setFormatter(formatter)

        self.token_logger.setLevel(logging.DEBUG)
        self.token_logger.addHandler(logger_handler)

    def new_session(self, session_id: int, name: str):
        """Create a new conversation if not exists"""
        if session_id not in self.memory:
            self.memory[session_id] = Conversation(name=name)
        self.memory[session_id] += Message(
            role=Role.SYS,
            content=self.__prompts['general']
        )


    def _get_assistant_index(self, user_input: str) -> int:
        """Determine assistant index based on user input

        :param user_input: The user's input query.
        :return: An index to choose the proper prompt.
        """
        route_messages = Conversation(
            name='get_assistant_index',
            messages=[
                {'role': 'system', 'content': self.__prompts['router']},
                {'role': 'user', 'content': user_input}
            ]
        )
        assistant_index_buffer = ''
        for chunk, _, _ in self.llm.query(route_messages):
            if not chunk:
                break
            assistant_index_buffer += chunk
        
        # Buscar el último número entre 1 y 3 en la respuesta
        match = re.search(r'[1-3](?=[^\d]*$)', assistant_index_buffer)
        if match:
            index = int(match.group(0))
            logger.debug(f"Extracted assistant index: {index} from response: {assistant_index_buffer}")
            return index
        
        # Si no se encuentra un número válido, loguear el error y devolver 1 por defecto
        logger.error(f'No valid assistant index found in: {assistant_index_buffer}')
        return 1
        
    def query(
        self,
        session_id: int,
        user_input: str
    ) -> Generator:
        """Handles the input from the user and generates responses in a
        streaming manner.

        :param session_id: The session identifier.
        :param user_input: The user's input query.

        :returns: Generator with response text in chunks."""
        # create a new conversation if not exists
        if not self.memory[session_id]:
            self.new_session(session_id)

        # route query
        router_result = self._get_assistant_index(user_input)
        logger.info(f"Router returned index: {router_result} for query: {user_input}")
        assistant_index = router_result  # Ensure we use the actual returned value
        logger.info(f"Using assistant index: {assistant_index} for query: {user_input}")
        

        # RESPONSE
        prompt = self.__prompts['general']
        user_input_with_tool_call = f'{user_input}'
        if assistant_index == 2:
            prompt = self.__prompts['reasoning']
        elif assistant_index == 3:
            tool_call_result = None
            tool_call_str = None
            logger.info(f"Executing tool call for query: {user_input}")
            for tool_call_execution in self.__tool_call(
                user_input,
                self.memory[session_id],
            ):
                tool_call_state = tool_call_execution['state']
                if tool_call_state == 'error':
                    logger.error(f"Tool call failed: {tool_call_execution.get('message', 'Unknown error')}")
                    break
                elif tool_call_state == 'running':
                    # should inform client of tool execution ...
                    tool_call_str = tool_call_execution['message']
                    logger.info(f"Tool call running: {tool_call_str}")
                else:
                    tool_call_result = tool_call_execution['message']
                    logger.info(f"Tool call completed with {len(tool_call_result) if tool_call_result else 0} characters of results")

            if tool_call_result:
                user_input_with_tool_call += (
                    f'\n### TOOL {tool_call_str} ###\n'
                    f'{tool_call_result}\n'
                    f'### TOOL {tool_call_str} END ###'
                )
                assistant_index = 1
            else:
                logger.warning(f"No tool call result for query: {user_input}")

        # Replace system prompt with the one built for specific assistant type
        conversation = self.memory[session_id]
        conversation.messages[0] = Message(role=Role.SYS, content=prompt)
        conversation += Message(
            role=Role.USER,
            content=user_input_with_tool_call
        )

        # note: conversation.message_dict doesn't care about context length
        response = ''
        # yes, I called ass_tokens the assistant tokens
        response_tokens = 0
        for chunk, usr_tokens, ass_tokens in self.llm.query(conversation):
            if usr_tokens:
                # set last message (usr) token usage
                conversation.messages[-1].set_tokens(usr_tokens)
                response_tokens = ass_tokens
                break
            if assistant_index == 1:
                response += chunk
                yield chunk
                continue

            for c in chunk:
                generation_state = self.__thought_parser.state(c)
                if generation_state == State.SPEAKING:
                    response += c
                    yield c
                # add thinking yield

        # remove tool call result from user input and add response to conversation
        conversation.messages[-1].content = user_input
        conversation += Message(
            role=Role.ASSISTANT,
            content=response,
        )

        conversation.messages[-1].set_tokens(response_tokens)
        logger.debug(f'CONVERSATION: {conversation}')

    def __should_use_direct_rag(self, user_input: str) -> bool:
        """Determines if a query should directly use RAG without the tool call mechanism."""
        # Keywords that suggest the user wants to search for information
        search_keywords = [
            "search", "find", "look up", "locate", "retrieve", 
            "tell me about", "what is", "how to", "explain",
            "information on", "details about"
        ]
        
        # Specific topics that should always use RAG
        topic_keywords = ["kerberoasting", "active-directory", "privilege-escalation"]
        
        # Check if query contains search intent or specific topics
        lower_input = user_input.lower()
        return (any(keyword in lower_input for keyword in search_keywords) or 
                any(topic in lower_input for topic in topic_keywords))

    def __handle_direct_rag_query(self, session_id: int, user_input: str) -> Generator:
        """Handles a query directly using RAG without the tool call mechanism."""
        from src.core.tools import RAG_SEARCH
        
        # Determine relevant topics based on query content
        topics = self.__extract_relevant_topics(user_input)
        topic_str = ",".join(topics) if topics else None
        
        logger.info(f"Executing direct RAG search with topics: {topic_str}")
        
        try:
            # Execute RAG search directly
            rag_result = RAG_SEARCH.run(
                rag_query=user_input,
                collection=None,
                topics=topic_str,
                collection_title=None,
                detail_level="detailed"
            )
            
            # Add user message to conversation
            conversation = self.memory[session_id]
            conversation += Message(role=Role.USER, content=user_input)
            
            # Format response with RAG results and yield it
            intro_text = "Based on my research, I've found the following information relevant to your query:\n\n"
            yield intro_text
            
            # Format and yield the RAG result
            formatted_result = self.__format_rag_result(rag_result)
            yield formatted_result
            
            # Add conclusion
            conclusion = "\n\nIs there anything specific about these findings you'd like me to elaborate on?"
            yield conclusion
            
            # Save the complete response to the conversation
            complete_response = intro_text + formatted_result + conclusion
            conversation += Message(role=Role.ASSISTANT, content=complete_response)
            
        except Exception as e:
            logger.error(f"Error in direct RAG query: {str(e)}")
            error_msg = "I apologize, but I encountered an issue while searching our knowledge base. "
            error_msg += "Let me answer based on my general knowledge instead.\n\n"
            
            yield error_msg
            
            # Fall back to general assistant
            conversation = self.memory[session_id]
            conversation += Message(role=Role.USER, content=user_input)
            conversation.messages[0] = Message(role=Role.SYS, content=self.__prompts['general'])
            
            # Generate a fallback response
            fallback_response = ""
            for chunk, _, _ in self.llm.query(conversation):
                if isinstance(chunk, str) and chunk:
                    fallback_response += chunk
                    yield chunk
            
            conversation += Message(role=Role.ASSISTANT, content=error_msg + fallback_response)

    def __extract_relevant_topics(self, query: str) -> List[str]:
        """Extract relevant topics from the query for RAG search."""
        topics = []
        query_lower = query.lower()
        
        # Map keywords to topics
        topic_mappings = {
            "active directory": ["active-directory", "domain-controllers"],
            "kerberos": ["active-directory", "kerberos", "authentication"],
            "sql": ["web-exploitation", "injection", "database"],
            "web": ["web-exploitation", "injection"],
            "password": ["credential-access", "authentication"],
            "credential": ["credential-access", "authentication"],
            "privilege": ["privilege-escalation"],
            "escalation": ["privilege-escalation"],
            "lateral": ["lateral-movement"],
            "evasion": ["defense-evasion"],
            "persistence": ["persistence"],
            "initial access": ["initial-access"],
            "phishing": ["initial-access", "social-engineering"],
            "reconnaissance": ["reconnaissance"],
            "enumeration": ["reconnaissance", "enumeration"],
            "cloud": ["cloud-exploitation"],
            "aws": ["cloud-exploitation"],
            "azure": ["cloud-exploitation"],
            "network": ["network-exploitation"],
            "wireless": ["wireless-exploitation"],
            "physical": ["physical"],
            "social": ["social-engineering"]
        }
        
        # Check for topic keywords in the query
        for keyword, related_topics in topic_mappings.items():
            if keyword in query_lower:
                topics.extend(related_topics)
        
        # Remove duplicates
        return list(set(topics))

    def __format_rag_result(self, result: str) -> str:
        """Format RAG results for better readability."""
        # Remove redundant section headers if present
        result = re.sub(r'From collection: ([^\n]+)\n-{60}', r'From collection: \1', result)
        
        # Add line breaks between different collections for better readability
        result = re.sub(r'From collection:', r'\n\nFrom collection:', result)
        
        # Remove any potential duplicate information
        lines = result.split('\n')
        unique_lines = []
        for line in lines:
            if line and line not in unique_lines:
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)

    def __tool_call(
        self,
        user_input: str,
        conversation: Conversation
    ) -> Generator:
        """Query a LLM for a tool call and executes it.

        :param user_input: The user's input query.
        :param conversation: The conversation history.

        :returns: Result of the tool execution."""
        # replace system prompt and generate tool call
        logger.info(f"Attempting tool call for query: {user_input}")
        
        original_system_prompt = conversation.messages[0].content
        conversation.messages[0] = Message(
            role='system',
            content=self.__prompts['tool']
        )
        conversation += Message(
            role='user',
            content=user_input
        )

        tool_call_response = ''
        for chunk, _, _ in self.llm.query(conversation):
            tool_call_response += chunk
        
        logger.info(f"Tool call response: {tool_call_response[:200]}...")
        
        # Restore original conversation state
        conversation.messages.pop()  # Remove the user message we added
        conversation.messages[0] = Message(role='system', content=original_system_prompt)

        # extract tool call and run it
        name, parameters, tool_extraction_error_message = self.__extract_tool_call(tool_call_response)
        if not name:
            error_message = f"Tool extraction failed: {tool_extraction_error_message}"
            logger.error(error_message)
            yield {
                'name': name,
                'parameters': parameters,
                'state': 'error',
                'message': error_message
            }
            return

        running_msg = (
            f"Running "
            f"{name.replace('_', ' ').capitalize()} "
            f"{list(parameters.values())[0]}"
        )
        logger.info(running_msg)
        yield {
            'name': name,
            'parameters': parameters,
            'state': 'running',
            'message': running_msg
        }
        try:
            tool_call_result = self.__tool_registry.compile(
                name=name,
                arguments=parameters
            )
            logger.info(f"Tool execution successful with {len(tool_call_result)} characters")
            yield {
                'name': name,
                'parameters': parameters,
                'state': 'done',
                'message': tool_call_result
            }
            return
        except Exception as tool_exec_error:
            error_message = (
                f'({type(tool_exec_error).__name__}): tool execution failed, '
                f'{tool_exec_error}'
            )
            logger.error(error_message)
            yield {
                'name': name,
                'parameters': parameters,
                'state': 'error',
                'message': error_message
            }
            return
        
    def __extract_tool_call(self, tool_call_response: str) -> Tuple[str | None, Dict, str | None]:
        """Extracts the tool call and its parameters from the model's response."""
        logger.info(f"Attempting to extract tool call from response: {tool_call_response[:200]}...")
        
        # Search for a JSON block containing "name" and "parameters"
        json_pattern = r'\{.*"name".*"parameters".*\}'
        match = re.search(json_pattern, tool_call_response, re.DOTALL)
        
        if match:
            try:
                tool_call_json = match.group(0)
                tool_call_dict = json.loads(tool_call_json)
                name = tool_call_dict.get("name")
                parameters = tool_call_dict.get("parameters", {})
                if name and parameters:
                    logger.info(f"Tool call extracted: {name} with parameters {parameters}")
                    return name, parameters, None
                else:
                    error_message = "Missing 'name' or 'parameters' in tool call"
                    logger.error(error_message)
                    return None, {}, error_message
            except json.JSONDecodeError as e:
                error_message = f"Could not parse JSON: {str(e)}"
                logger.error(error_message)
                return None, {}, error_message
        else:
            error_message = "No valid tool call found in JSON format"
            logger.error(error_message)
            return None, {}, error_message
