import re
import json
import logging
from typing import Generator, Dict, Tuple

from tool_parse import ToolRegistry
from src.agent import AgentArchitecture
from src.core import Conversation
from src.core.llm import LLM
from src.core.memory import Message, Role
from src.utils import get_logger, LOGS_PATH
from src.core.knowledge.store import QdrantStore, rag_store

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
                    if self.__count == 2:
                        self.__count = 0
                        self.__state = State.THINKING
                        return State.THINKING
                    self.__count += 1
                    return State.IDLE
                else:
                    self.__count = 0
                    return State.SPEAKING
            case State.THINKING:
                if c == '@':
                    if self.__count == 2:
                        self.__count = 0
                        self.__state = State.SPEAKING
                        return State.IDLE
                    self.__count += 1
                    return State.IDLE
                else:
                    self.__count = 0
                    return State.THINKING

class DefaultArchitecture(AgentArchitecture):
    """
    Default architecture for the AI-OPS agent, integrating RAG with HybridRetriever.
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

        # Inicialización del HybridRetriever desde rag_store
        try:
            self.hybrid_retriever = rag_store.hybrid_retriever
            logger.info("Using rag_store.hybrid_retriever for RAG operations")
        except Exception as e:
            logger.error(f"Error accessing rag_store.hybrid_retriever: {str(e)}")
            self.hybrid_retriever = None

        self.__thought_parser: State = State()
        self.__tool_pattern = r"\s*({[^}]*(?:{[^}]*})*[^}]*}|\[[^\]]*(?:\[[^\]]*\])*[^\]]*\])\s*$"
        tool_names = ', '.join([tool["function"]["name"] for tool in self.__tools])
        logger.info(f'Initialized DefaultArchitecture with model {llm.model} and tools {tool_names}')

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

    def query(self, session_id: int, user_input: str) -> Generator:
        """Handles user input and generates responses, enriched with RAG if available."""
        if not self.memory[session_id]:
            self.new_session(session_id)

        assistant_index = self.__get_assistant_index(user_input)
        prompt = self.__prompts['general']
        user_input_with_tool_call = user_input

        # Manejo de prompts según el tipo de consulta
        if assistant_index == 2:
            prompt = self.__prompts['reasoning']
        elif assistant_index == 3:
            tool_call_result: str | None = None
            tool_call_str: str | None = None
            for tool_call_execution in self.__tool_call(user_input, self.memory[session_id]):
                tool_call_state = tool_call_execution['state']
                if tool_call_state == 'error':
                    break
                elif tool_call_state == 'running':
                    tool_call_str = tool_call_execution['message']
                else:
                    tool_call_result = tool_call_execution['message']
            if tool_call_result:
                user_input_with_tool_call += (
                    f'\n### TOOL {tool_call_str} ###\n'
                    f'{tool_call_result}\n'
                    f'### TOOL {tool_call_str} END ###'
                )
                assistant_index = 1

        conversation = self.memory[session_id]
        conversation.messages[0] = Message(role=Role.SYS, content=prompt)
        conversation += Message(role=Role.USER, content=user_input_with_tool_call)

        # Enriquecimiento con RAG usando rag_store.hybrid_retriever
        if rag_store.hybrid_retriever is not None:
            try:
                retrieved_results = rag_store.hybrid_retriever.retrieve(query=user_input, top_k=5, use_graph=True)
                if retrieved_results:
                    context = "\n\n".join([
                        f"[Document: {result['metadata']['source']}]\n{result['text']}"
                        for result in retrieved_results
                    ])
                    augmented_input = (
                        f"Next information can be useful to answer:\n\n"
                        f"{context}\n\n"
                        f"Question from user:\n{user_input_with_tool_call}"
                    )
                    conversation.messages[-1].content = augmented_input
                    logger.info(f"Query enriched with {len(retrieved_results)} document chunks")
                else:
                    logger.info("No relevant documents found for the query.")
            except Exception as e:
                logger.error(f"Error during RAG retrieval: {str(e)}")
        else:
            logger.warning("HybridRetriever not initialized in rag_store")

        # Generación de la respuesta
        response = ''
        response_tokens = 0
        for chunk, usr_tokens, ass_tokens in self.llm.query(conversation):
            if usr_tokens:
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

        conversation.messages[-1].content = user_input
        conversation += Message(role=Role.ASSISTANT, content=response)
        conversation.messages[-1].set_tokens(response_tokens)
        logger.debug(f'CONVERSATION: {conversation}')

    def new_session(self, session_id: int, name: str):
        """Create a new conversation if it doesn't exist."""
        if session_id not in self.memory:
            self.memory[session_id] = Conversation(name=name)
        self.memory[session_id] += Message(role=Role.SYS, content=self.__prompts['general'])

    def __get_assistant_index(self, user_input: str) -> int:
        """Determine assistant index based on user input."""
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
        try:
            return int(assistant_index_buffer.strip()[:1])
        except ValueError:
            logger.error(f'Wrong assistant index: {assistant_index_buffer}')
            return 1

    def __tool_call(self, user_input: str, conversation: Conversation) -> Generator[Dict, None, None]:
        """Query an LLM for a tool call and execute it."""
        conversation.messages[0] = Message(role='system', content=self.__prompts['tool'])
        conversation += Message(role='user', content=user_input)
        tool_call_response = ''
        for chunk, _, _ in self.llm.query(conversation):
            tool_call_response += chunk
        logger.debug(f"Tool call response: {tool_call_response}")
        name, parameters, tool_extraction_error_message = self.__extract_tool_call(tool_call_response)
        if not name:
            logger.warning(f"Tool extraction failed: {tool_extraction_error_message}")
            yield {'name': name, 'parameters': parameters, 'state': 'error', 'message': tool_extraction_error_message}
            return
        running_msg = f"Running {name.replace('_', ' ').capitalize()} {list(parameters.values())[0] if parameters else 'with no parameters'}"
        logger.info(running_msg)
        yield {'name': name, 'parameters': parameters, 'state': 'running', 'message': running_msg}
        try:
            tool_call_result = self.__tool_registry.compile(name=name, arguments=parameters)
            logger.info(f"Tool result length: {len(tool_call_result) if tool_call_result else 0} characters")
            yield {'name': name, 'parameters': parameters, 'state': 'done', 'message': tool_call_result}
        except Exception as tool_exec_error:
            error_message = f'({type(tool_exec_error).__name__}): tool execution failed, {tool_exec_error}'
            logger.error(error_message)
            yield {'name': name, 'parameters': parameters, 'state': 'error', 'message': error_message}

    def __extract_tool_call(self, tool_call_response: str) -> Tuple[str | None, Dict, str | None]:
        """Extract tool call and parameters from LLM response."""
        tool_call_match = re.search(self.__tool_pattern, tool_call_response)
        if not tool_call_match:
            error_message = f'Tool call failed: not found in LLM response: {tool_call_response}'
            logger.error(error_message)
            return None, {}, error_message
        try:
            tool_call_json = tool_call_match.group(1).replace("'", '"').strip()
            tool_call_dict = json.loads(tool_call_json)
            name, parameters = tool_call_dict['name'], tool_call_dict['parameters']
        except json.JSONDecodeError as json_extract_err:
            error_message = f'Tool call failed: not found in LLM response: {tool_call_response}\nError: {json_extract_err}'
            logger.error(error_message)
            return None, {}, error_message
        found = any(t['function']['name'] == name for t in self.__tools)
        if not found:
            error_message = f'Tool call failed: {name} is not a tool.'
            logger.error(error_message)
            return None, {}, error_message
        return name, parameters, None