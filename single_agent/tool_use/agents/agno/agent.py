"""
Agno Agent Implementation

Implements BaseAgent interface using Agno framework for tool-calling agents.
Uses Agno's Agent class which automatically converts Python functions to tools.

Important: Tools are pre-selected by MASBench ToolSelector.
Agno does not perform additional filtering - it uses whatever tools are provided.

Note on Tool Limits:
- OpenAI has a limit of 128 tools per request
- Agno's Agent sends ALL bound tools to the LLM in each request
- Agno has no built-in tool filtering mechanism
- This is why centralized tool selection (ToolSelector) is essential for Agno

TOOL EXECUTION TRACKING:
Agno detects tool calls in model outputs and executes them automatically within
its native runtime. Our adapter tracks tool executions by wrapping the tool
functions and recording when they are called by Agno.

The wrapper functions:
- Are called by Agno when tools are executed
- Call the original LangChain tool functions (which call StableToolBench server)
- Track executions in self._executed_tools for API scoring
- Return results to Agno for continued processing

This ensures that tool executions are properly tracked and reported in the benchmark.
"""
import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from inspect import Parameter

# Import Agno components
try:
    from agno.agent import Agent as AgnoAgent
    from agno.models.openai import OpenAIChat
    AGNO_AVAILABLE = True
    AGNO_IMPORT_ERROR = None
except ImportError as e:
    AGNO_AVAILABLE = False
    AGNO_IMPORT_ERROR = e
    # Create dummy classes to allow module import
    class AgnoAgent:
        pass
    class OpenAIChat:
        pass

# Check if LangChain tools are being passed (from run_benchmark.py)
try:
    from langchain_core.tools import BaseTool as LangChainBaseTool, StructuredTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LangChainBaseTool = None
    StructuredTool = None

from ..base_agent import BaseAgent
from utils.tool_utils import create_tool_function, sanitize_tool_name


class AgnoAgentClass(BaseAgent):
    """
    Agno agent that implements BaseAgent interface.
    
    Uses Agno's Agent which:
    1. Uses a language model with tool calling capability
    2. Automatically converts Python functions to tools
    3. Executes tools and returns results
    4. Supports async execution
    
    This agent:
    1. Loads tools from StableToolBench (via LangChain loader in run_benchmark.py)
    2. Converts LangChain tools to Python functions for Agno
    3. Uses LLM with tool calling via Agno Agent
    4. Returns answers in StableToolBench format with ExecutionGraph format
    
    Note: Requires Agno to be installed. If Agno is not available,
    this class cannot be instantiated (raises ImportError).
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        server_url: str = "http://localhost:8080/virtual",
        temperature: float = 0.0,
        verbose: bool = False,
        max_tools: int = 120
    ):
        """
        Initialize Agno agent.
        
        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            server_url: URL of the server for tool calls
            temperature: Temperature for LLM (default: 0.0)
            verbose: Whether to print debug information
            max_tools: Maximum number of tools to use per query (default: 120, OpenAI limit is 128)
        """
        if not AGNO_AVAILABLE:
            error_msg = (
                "Agno is not installed. "
                "Install it with:\n"
                "  pip install agno"
            )
            if AGNO_IMPORT_ERROR is not None:
                raise ImportError(error_msg) from AGNO_IMPORT_ERROR
            else:
                raise ImportError(error_msg)
        
        self.model = model
        self.server_url = server_url
        self.temperature = temperature
        self.verbose = verbose
        self.max_tools = max_tools
        
        # Tools are pre-selected by ToolSelector and bound here
        # Agno does not perform additional filtering
        self.bound_tools: List[Callable] = []  # Python functions for Agno
        self.tool_metadata: Dict[str, Dict[str, str]] = {}  # Map tool name to metadata
        self.tools_bound = False
        # Track tool executions per query (reset in answer())
        self._executed_tools: List[List[str]] = []  # List of [tool_name, api_name] pairs
        # Note: We don't store the agent instance - we create it fresh per query
        # to avoid state leakage between queries (similar to LangGraph)
    
    def bind_tools(
        self,
        tools: Union[List[LangChainBaseTool], str, None] = None,
        tools_dir: Optional[str] = None,
        server_url: str = "http://localhost:8080/virtual"
    ) -> None:
        """
        Bind tools to the agent.
        
        Args:
            tools: Pre-selected list of tools (preferred - from ToolSelector)
            tools_dir: Path to StableToolBench/toolenv/tools/ directory (legacy - loads all tools)
            server_url: URL of the server for tool calls
        """
        # Update server URL if provided
        self.server_url = server_url
        
        if tools is not None:
            # Use pre-selected tools (preferred approach - from ToolSelector)
            if isinstance(tools, list):
                # Check if tools are LangChain tools (from run_benchmark.py)
                # If so, convert them to Agno-compatible Python functions
                if len(tools) > 0:
                    first_tool = tools[0]
                    if LANGCHAIN_AVAILABLE and isinstance(first_tool, StructuredTool):
                        # Convert LangChain tools to Agno-compatible Python functions
                        if self.verbose:
                            print(f"[AgnoAgent] Converting {len(tools)} LangChain tools to Agno tools...")
                        self.bound_tools = self._convert_langchain_to_agno_tools(tools)
                    else:
                        # Assume they're already Python functions
                        self.bound_tools = tools
                else:
                    self.bound_tools = []
                
                # Fail loudly if tool count exceeds max_tools (benchmark integrity)
                if len(self.bound_tools) > self.max_tools:
                    raise RuntimeError(
                        f"ToolSelector returned {len(self.bound_tools)} tools, "
                        f"exceeds max_tools={self.max_tools}. "
                        f"This violates benchmark integrity."
                    )
                
                if self.verbose:
                    print(f"[AgnoAgent] Bound {len(self.bound_tools)} pre-selected tools")
            else:
                raise ValueError("tools must be a list of BaseTool objects or Python functions")
        elif tools_dir is not None:
            # Legacy mode: load all tools from directory (not recommended for benchmarks)
            raise NotImplementedError(
                "Direct tool loading from directory is not implemented for Agno. "
                "Use the centralized ToolSelector via run_benchmark.py instead."
            )
        else:
            raise ValueError("Either 'tools' or 'tools_dir' must be provided")
        
        # Note: We don't create the agent here because Agno agents may be stateful
        # We'll create it fresh for each query in answer() to avoid state leakage
        self.tools_bound = True
    
    def _convert_langchain_to_agno_tools(self, langchain_tools: List) -> List[Callable]:
        """
        Convert LangChain tools to Agno-compatible Python functions.
        
        Agno automatically converts Python functions to tools, so we need to:
        1. Extract tool name, description, and parameters from LangChain tool
        2. Create a Python function with proper type annotations
        3. Make the function call the original LangChain tool function
        
        Args:
            langchain_tools: List of LangChain StructuredTool objects
            
        Returns:
            List of Python functions that Agno can use as tools
        """
        agno_tools = []
        
        for langchain_tool in langchain_tools:
            # Extract metadata from LangChain tool
            tool_name_raw = langchain_tool.name
            # Sanitize tool name for Python function name (remove special chars, ensure valid identifier)
            tool_name = sanitize_tool_name(tool_name_raw, max_length=64).replace('-', '_')
            # Remove any remaining consecutive underscores
            while '__' in tool_name:
                tool_name = tool_name.replace('__', '_')
            description = langchain_tool.description
            
            # Get structured metadata if available
            metadata = getattr(langchain_tool, 'metadata', {}) or {}
            tool_name_original = metadata.get('tool_name', '')
            api_name = metadata.get('api_name', '')
            category = metadata.get('category', '')
            original_name = metadata.get('original_name', tool_name)
            
            # Get the original tool function from LangChain tool
            original_func = langchain_tool.func
            
            # Get args_schema from LangChain tool to extract parameter information
            args_schema = langchain_tool.args_schema
            
            # Create a Python function that Agno can use
            # Agno will automatically extract the signature and docstring
            if args_schema and hasattr(args_schema, 'model_fields'):
                # Extract parameters from Pydantic model to create proper function signature
                from typing import get_type_hints
                
                # Build function signature from schema
                param_defs = []
                for field_name, field_info in args_schema.model_fields.items():
                    # Get field type and unwrap Optional/Union
                    field_type = field_info.annotation
                    if field_type is None or field_type == Parameter.empty:
                        field_type = str
                    
                    # Unwrap Optional[T] or Union[T, None] to get the actual type
                    origin = getattr(field_type, '__origin__', None)
                    if origin is Union:
                        # Get the non-None type from Union[T, None] or Union[None, T]
                        args = getattr(field_type, '__args__', ())
                        non_none_types = [arg for arg in args if arg is not type(None)]
                        if non_none_types:
                            field_type = non_none_types[0]  # Use first non-None type
                        else:
                            field_type = str  # Fallback if all are None
                    elif hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                        # Handle typing.Union (Python < 3.10)
                        args = getattr(field_type, '__args__', ())
                        non_none_types = [arg for arg in args if arg is not type(None)]
                        if non_none_types:
                            field_type = non_none_types[0]
                        else:
                            field_type = str
                    
                    # Check if field has default
                    default_value = Parameter.empty
                    if hasattr(field_info, 'default'):
                        default_value = field_info.default
                    
                    # Check for PydanticUndefined
                    try:
                        from pydantic_core import PydanticUndefined as PUndef
                    except ImportError:
                        try:
                            from pydantic import PydanticUndefined as PUndef
                        except ImportError:
                            PUndef = type('PydanticUndefined', (), {})
                    
                    if (default_value is not Parameter.empty and 
                        default_value is not None and
                        not isinstance(default_value, type(PUndef)) and
                        default_value is not PUndef):
                        param_defs.append((field_name, field_type, default_value))
                    else:
                        param_defs.append((field_name, field_type, None))
                
                # Create function with proper signature using exec
                # This is necessary because Agno needs explicit parameters
                # Store metadata for tracking (capture in closure)
                tool_name_original = metadata.get('tool_name', tool_name)
                api_name = metadata.get('api_name', '')
                agent_self = self  # Capture self in closure
                
                def make_typed_function(param_defs, original_func_ref, desc, tool_name, tool_name_orig, api_name_val, agent_instance):
                    # Build function signature string
                    param_strs = []
                    for param_name, param_type, default_val in param_defs:
                        type_str = param_type.__name__ if hasattr(param_type, '__name__') else 'str'
                        if default_val is not None and default_val is not Parameter.empty:
                            # Skip PydanticUndefined
                            try:
                                from pydantic_core import PydanticUndefined as PUndef
                            except ImportError:
                                try:
                                    from pydantic import PydanticUndefined as PUndef
                                except ImportError:
                                    PUndef = type('PydanticUndefined', (), {})
                            
                            if isinstance(default_val, type(PUndef)) or default_val is PUndef:
                                param_strs.append(f"{param_name}: {type_str}")
                            elif isinstance(default_val, str):
                                escaped_val = default_val.replace("'", "\\'")
                                param_strs.append(f"{param_name}: {type_str} = '{escaped_val}'")
                            elif isinstance(default_val, (int, float, bool)):
                                param_strs.append(f"{param_name}: {type_str} = {default_val}")
                            elif default_val is None:
                                # For None defaults, use Any to avoid type issues
                                param_strs.append(f"{param_name}: Any = None")
                            else:
                                param_strs.append(f"{param_name}: {type_str} = {repr(default_val)}")
                        else:
                            param_strs.append(f"{param_name}: {type_str}")
                    
                    param_str = ', '.join(param_strs)
                    
                    # Build function body
                    param_names = [p[0] for p in param_defs]
                    kwargs_build = ', '.join([f"'{p}': {p}" for p in param_names])
                    
                    # Use repr() to safely embed strings with apostrophes and special characters
                    tool_name_orig_repr = repr(tool_name_orig)
                    api_name_val_repr = repr(api_name_val)
                    
                    # Capture variables - use default arguments to ensure they're available at runtime
                    # Default arguments are evaluated at function definition time, so they capture the values
                    captured_agent = agent_instance
                    captured_func = original_func_ref
                    
                    # Build function body - use default arguments to access captured variables
                    # Add imports at the start of the function body so modules are available at runtime
                    func_body = f"""
    import asyncio
    import json
    
    # Track tool execution (using variables from default arguments)
    if hasattr(_captured_agent, '_executed_tools') and _captured_agent._executed_tools is not None:
        if {tool_name_orig_repr} and {api_name_val_repr}:
            _captured_agent._executed_tools.append([{tool_name_orig_repr}, {api_name_val_repr}])
    
    try:
        kwargs = {{{kwargs_build}}}
        if asyncio.iscoroutinefunction(_captured_func):
            result = await _captured_func(**kwargs)
        else:
            result = _captured_func(**kwargs)
        if isinstance(result, dict):
            return json.dumps(result)
        return str(result)
    except Exception as e:
        return json.dumps({{"error": str(e), "response": ""}})
"""
                    
                    # Add hidden default arguments to capture the variables
                    # These will be evaluated at function definition time
                    if param_str:
                        param_str_with_defaults = f"{param_str}, _captured_agent=captured_agent, _captured_func=captured_func"
                    else:
                        param_str_with_defaults = f"_captured_agent=captured_agent, _captured_func=captured_func"
                    
                    # Create function dynamically
                    desc_escaped = desc.replace('"""', "'''")
                    func_code = f'async def {tool_name}({param_str_with_defaults}) -> str:\n    """{desc_escaped}"""\n{func_body}'
                    
                    # Create local_vars - captured_agent and captured_func for default arg evaluation
                    local_vars = {
                        'captured_agent': captured_agent,  # For default arg evaluation
                        'captured_func': captured_func,  # For default arg evaluation
                        'asyncio': asyncio,
                        'json': json,
                        'isinstance': isinstance,
                        'str': str,
                        'dict': dict,
                        'Any': Any,
                        'Parameter': Parameter
                    }
                    
                    # Execute - default args will capture the values at definition time
                    exec(func_code, {'__builtins__': __builtins__}, local_vars)
                    wrapper_func = local_vars[tool_name]
                    
                    # The function now has default args that capture the values
                    # We need to hide these from Agno's signature inspection but keep them functional
                    # The simplest approach: return the function directly, but modify its signature
                    from inspect import signature as sig
                    
                    # Get the original signature
                    orig_sig = sig(wrapper_func)
                    # Filter out the hidden args for the visible signature
                    visible_params = [p for p in orig_sig.parameters.values() if not p.name.startswith('_')]
                    visible_sig = orig_sig.replace(parameters=visible_params)
                    
                    # Set the visible signature (without hidden args)
                    # The default args are still there and will be used when Agno calls the function
                    wrapper_func.__signature__ = visible_sig
                    
                    return wrapper_func
                
                wrapper_func = make_typed_function(
                    param_defs, original_func, description, tool_name,
                    tool_name_original, api_name, agent_self
                )
            else:
                # Fallback: create simple wrapper with execution tracking
                tool_name_original = metadata.get('tool_name', tool_name)
                api_name = metadata.get('api_name', '')
                
                # Create a closure that captures self and other variables
                captured_self = self
                captured_original_func = original_func
                captured_tool_name_orig = tool_name_original
                captured_api_name = api_name
                
                async def wrapper_func(**kwargs) -> str:
                    """Wrapper function for LangChain tool."""
                    # Track tool execution (using captured variables)
                    if hasattr(captured_self, '_executed_tools') and captured_self._executed_tools is not None:
                        if captured_tool_name_orig and captured_api_name:
                            captured_self._executed_tools.append([captured_tool_name_orig, captured_api_name])
                    
                    try:
                        if asyncio.iscoroutinefunction(captured_original_func):
                            result = await captured_original_func(**kwargs)
                        else:
                            result = captured_original_func(**kwargs)
                        if isinstance(result, dict):
                            return json.dumps(result)
                        return str(result)
                    except Exception as e:
                        return json.dumps({"error": str(e), "response": ""})
                
                wrapper_func.__name__ = tool_name
                wrapper_func.__doc__ = description
            
            # Store metadata for later extraction
            self.tool_metadata[tool_name] = {
                'tool_name': tool_name_original or tool_name,
                'api_name': api_name,
                'category': category,
                'original_name': original_name
            }
            
            agno_tools.append(wrapper_func)
        
        return agno_tools
    
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for a query.
        
        Args:
            query: The query string to answer
            
        Returns:
            Dictionary in StableToolBench format with answer and called_apis
        """
        if not self.tools_bound:
            raise ValueError("Tools must be bound before calling answer(). Call bind_tools() first.")
        
        if self.verbose:
            print(f"[AgnoAgent] Processing query: {query[:100]}...")
        
        # Agno uses async, so we need to run the async code
        return asyncio.run(self._answer_async(query))
    
    async def _answer_async(self, query: str) -> Dict[str, Any]:
        """
        Async implementation of answer().
        
        Args:
            query: The query string to answer
            
        Returns:
            Dictionary in StableToolBench format with answer and called_apis
        """
        if not self.tools_bound:
            raise ValueError("Tools must be bound before calling answer(). Call bind_tools() first.")
        
        # Reset execution tracking for this query
        self._executed_tools = []
        
        if self.verbose:
            print(f"[AgnoAgent] Using {len(self.bound_tools)} tools")
        
        # Create Agno agent fresh for each query to avoid state leakage
        # Similar to LangGraph, we recreate the agent per query
        agno_agent = AgnoAgent(
            model=OpenAIChat(id=self.model, temperature=self.temperature),
            tools=self.bound_tools,
            markdown=False,  # We want plain text for StableToolBench
        )
        
        # Execute agent (async)
        try:
            if self.verbose:
                print(f"[AgnoAgent] Executing query with Agno agent...")
            # Use arun to get the response
            response = await agno_agent.arun(query)
            if self.verbose:
                print(f"[AgnoAgent] Received response, type: {type(response)}")
                # Debug: print response structure
                if hasattr(response, '__dict__'):
                    print(f"[AgnoAgent] Response attributes: {list(response.__dict__.keys())}")
        except Exception as e:
            if self.verbose:
                print(f"[AgnoAgent] Error during execution: {e}")
                import traceback
                traceback.print_exc()
            raise
        
        # Extract called_apis and build answer_details from response
        # First, check if tools were actually executed (via our tracking)
        called_apis: List[List[str]] = self._executed_tools.copy() if self._executed_tools else []
        answer_details: List[Dict[str, Any]] = []
        final_answer = ""
        
        if self.verbose:
            print(f"[AgnoAgent] Tools executed (via tracking): {len(called_apis)}")
        
        # Extract final answer from response
        # Agno's response structure may vary - check common attributes
        if self.verbose:
            print(f"[AgnoAgent] Inspecting response structure...")
            print(f"[AgnoAgent] Response type: {type(response)}")
            if hasattr(response, '__dict__'):
                print(f"[AgnoAgent] Response attributes: {list(response.__dict__.keys())[:10]}")  # First 10 attributes
        
        if hasattr(response, 'content'):
            final_answer = str(response.content)
        elif hasattr(response, 'text'):
            final_answer = str(response.text)
        elif isinstance(response, str):
            final_answer = response
        elif hasattr(response, 'messages') and response.messages:
            # Try to get last message
            if self.verbose:
                print(f"[AgnoAgent] Found {len(response.messages)} messages")
            last_message = response.messages[-1]
            if hasattr(last_message, 'content'):
                final_answer = str(last_message.content)
            elif hasattr(last_message, 'text'):
                final_answer = str(last_message.text)
        else:
            # Fallback: try to convert response to string
            final_answer = str(response)
        
        if self.verbose:
            print(f"[AgnoAgent] Extracted final answer (length: {len(final_answer)})")
        
        # Extract tool calls from response
        # Agno stores tool calls in the response - need to check multiple possible structures
        # Try different common patterns for tool call storage
        
        # Pattern 1: Check response.messages for tool calls
        detected_tool_calls_count = 0
        if hasattr(response, 'messages') and response.messages is not None:
            if self.verbose:
                print(f"[AgnoAgent] Checking {len(response.messages)} messages for tool calls")
            for i, message in enumerate(response.messages):
                if self.verbose and i < 3:  # Debug first 3 messages
                    has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
                    print(f"[AgnoAgent] Message {i}: type={type(message)}, has tool_calls={has_tool_calls}")
                    if has_tool_calls:
                        print(f"[AgnoAgent] Detected {len(message.tool_calls)} tool calls in message {i}")
                        print(f"[AgnoAgent] Tool executions are tracked via wrapper functions")
                # Check if message contains tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    detected_tool_calls_count += len(message.tool_calls)
                    if self.verbose:
                        print(f"[AgnoAgent] Found {len(message.tool_calls)} tool calls in message {i}")
                    for tool_call in message.tool_calls:
                        # Extract tool name and arguments
                        tool_name = getattr(tool_call, 'name', None)
                        if not tool_name and hasattr(tool_call, 'function'):
                            func = tool_call.function
                            tool_name = getattr(func, 'name', None) if hasattr(func, 'name') else (func.get('name', '') if isinstance(func, dict) else '')
                        
                        arguments = getattr(tool_call, 'arguments', None)
                        if not arguments and hasattr(tool_call, 'function'):
                            func = tool_call.function
                            arguments = getattr(func, 'arguments', None) if hasattr(func, 'arguments') else (func.get('arguments', {}) if isinstance(func, dict) else {})
                        
                        output = getattr(tool_call, 'output', None) or getattr(tool_call, 'result', None)
                        
                        if tool_name:
                            # Get metadata for this tool (try sanitized name first, then original)
                            metadata = self.tool_metadata.get(tool_name, {})
                            if not metadata:
                                # Try to find by matching sanitized name
                                for stored_name, stored_meta in self.tool_metadata.items():
                                    if stored_name.replace('_', '-') == tool_name.replace('_', '-') or stored_meta.get('original_name', '') == tool_name:
                                        metadata = stored_meta
                                        break
                            
                            tool_name_original = metadata.get('tool_name', tool_name) if metadata else tool_name
                            api_name = metadata.get('api_name', '') if metadata else ''
                            
                            # Record API call
                            if tool_name_original and api_name:
                                called_apis.append([tool_name_original, api_name])
                            
                            # Parse arguments if needed
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except:
                                    arguments = {"raw": arguments}
                            elif arguments is None:
                                arguments = {}
                            
                            # Add to answer_details
                            tool_call_detail = {
                                "role": "tool",
                                "message": json.dumps({
                                    "name": f"{tool_name_original}_{api_name}" if api_name else tool_name_original,
                                    "arguments": arguments,
                                    "response": str(output) if output else ""
                                }),
                                "next": []
                            }
                            answer_details.append(tool_call_detail)
        
        # Pattern 2: Check response.run for tool calls
        if not called_apis and hasattr(response, 'run'):
            run = response.run
            if hasattr(run, 'tool_calls') and run.tool_calls:
                for tool_call in run.tool_calls:
                    tool_name = getattr(tool_call, 'name', None)
                    if not tool_name and hasattr(tool_call, 'function'):
                        func = tool_call.function
                        tool_name = getattr(func, 'name', None) if hasattr(func, 'name') else (func.get('name', '') if isinstance(func, dict) else '')
                    
                    if tool_name:
                        metadata = self.tool_metadata.get(tool_name, {})
                        if not metadata:
                            for stored_name, stored_meta in self.tool_metadata.items():
                                if stored_name.replace('_', '-') == tool_name.replace('_', '-') or stored_meta.get('original_name', '') == tool_name:
                                    metadata = stored_meta
                                    break
                        
                        tool_name_original = metadata.get('tool_name', tool_name) if metadata else tool_name
                        api_name = metadata.get('api_name', '') if metadata else ''
                        if tool_name_original and api_name:
                            called_apis.append([tool_name_original, api_name])
        
        # Pattern 3: Check response directly for tool_calls attribute
        if not called_apis and hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call, 'name', None) or (tool_call.get('name', '') if isinstance(tool_call, dict) else '')
                if tool_name:
                    metadata = self.tool_metadata.get(tool_name, {})
                    if not metadata:
                        for stored_name, stored_meta in self.tool_metadata.items():
                            if stored_name.replace('_', '-') == tool_name.replace('_', '-') or stored_meta.get('original_name', '') == tool_name:
                                metadata = stored_meta
                                break
                    
                    tool_name_original = metadata.get('tool_name', tool_name) if metadata else tool_name
                    api_name = metadata.get('api_name', '') if metadata else ''
                    if tool_name_original and api_name:
                        called_apis.append([tool_name_original, api_name])
        
        # Add Finish call to answer_details
        finish_detail = {
            "role": "tool",
            "message": json.dumps({
                "name": "Finish",
                "arguments": {
                    "return_type": "give_answer",
                    "final_answer": final_answer
                },
                "response": ""
            }),
            "next": []
        }
        answer_details.append(finish_detail)
        
        if self.verbose:
            print(f"[AgnoAgent] Generated answer (executed tool calls: {len(called_apis)})")
            if detected_tool_calls_count > 0 and len(called_apis) == 0:
                print(f"[AgnoAgent] ⚠️  WARNING: Detected {detected_tool_calls_count} tool calls in messages, but 0 were executed")
                print(f"[AgnoAgent] ⚠️  This may indicate that Agno detected tool calls but did not execute them")
                print(f"[AgnoAgent] ⚠️  or that tool execution tracking is not working correctly")
        
        # Return in StableToolBench format
        # called_apis is populated from execution tracking in wrapper functions
        return {
            "answer": {
                "final_answer": final_answer,
                "answer_details": answer_details
            },
            "called_apis": called_apis  # Will be empty - Agno doesn't execute tools automatically
        }

