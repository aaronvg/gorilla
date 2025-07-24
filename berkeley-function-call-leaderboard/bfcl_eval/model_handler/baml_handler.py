import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import (
    func_doc_language_specific_pre_processing,
    convert_to_tool,
)
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.baml_utils import (
    detect_baml_provider,
)

try:
    from bfcl_eval.baml_client import b
    from bfcl_eval.baml_client.type_builder import TypeBuilder
    from baml_py import ClientRegistry
except ImportError:
    raise ImportError(
        "BAML is required when using --tool-format baml. "
        "Please ensure BAML client is properly configured and generated."
    )


class BAMLHandler(BaseHandler):
    def __init__(self, model_name: str, temperature: float = 0.0) -> None:
        # Modify model name to include BAML suffix to avoid result collision
        baml_model_name = f"{model_name}-baml"
        super().__init__(baml_model_name, temperature)

        # Store original model name for API calls
        self.original_model_name = model_name
        self.model_style = ModelStyle.OpenAI_Completions  # Default style
        self.is_fc_model = True  # BAML always works in FC mode
        self.underscore_to_dot = (
            True  # Default value, will be overridden by eval_runner
        )
        self.base_url = self._get_base_url_from_model_name()

    def _get_base_url_from_model_name(self) -> Optional[str]:
        """Extract base_url if needed for custom models."""
        # This can be extended to read from config if needed
        return None

    def _create_client_registry(self, **kwargs) -> ClientRegistry:
        """Create BAML client with auto-detected provider."""
        cr = ClientRegistry()
        provider, provider_options = detect_baml_provider(
            self.original_model_name, self.base_url
        )

        # Get API key based on provider
        api_key = self._get_api_key(provider)

        # Merge provider options with common options
        options = {
            "model": self.original_model_name,  # Use original model name for API calls
            "api_key": api_key,
            "temperature": self.temperature,
            **provider_options,
            **kwargs,  # Allow override of any options
        }

        cr.add_llm_client(name="DynamicClient", provider=provider, options=options)
        cr.set_primary("DynamicClient")
        return cr

    def _convert_param_value(self, param_value: str):
        """Convert string parameter values to appropriate Python types."""
        # Handle None
        if param_value == "None":
            return None

        # Handle boolean values
        if param_value in ("True", "true"):
            return True
        elif param_value in ("False", "false"):
            return False

        # Handle numbers (including scientific notation)
        try:
            # Try scientific notation first
            if "e" in param_value.lower():
                return float(param_value)
            # Try float
            elif "." in param_value:
                return float(param_value)
            # Try int
            else:
                return int(param_value)
        except ValueError:
            # Not a number, keep as string
            pass

        # Handle arrays (basic parsing)
        if param_value.startswith("[") and param_value.endswith("]"):
            try:
                # Use json.loads for proper array parsing
                return json.loads(param_value)
            except json.JSONDecodeError:
                # If JSON parsing fails, keep as string
                pass

        # Keep as string by default
        return param_value

    def _get_api_key(self, provider: str) -> str:
        """Get the appropriate API key for the provider."""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "openai-responses": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google-ai": "GOOGLE_API_KEY",
            "openai-generic": "OPENAI_API_KEY",  # Most generic APIs use this format
        }

        env_key = key_mapping.get(provider)
        if env_key:
            api_key = os.getenv(env_key)
            if not api_key:
                raise ValueError(
                    f"Missing environment variable {env_key} for provider {provider}"
                )
            return api_key

        # For AWS Bedrock, credentials are handled by AWS SDK
        if provider == "aws-bedrock":
            return ""

        raise ValueError(f"Unknown provider: {provider}")

    def _get_baml_field_type(
        self, schema: Dict[str, Any], tb: TypeBuilder, required: bool = True
    ):
        """Convert JSON schema to BAML FieldType using TypeBuilder APIs."""
        schema_type = schema.get("type", "string")

        if schema_type == "string":
            if "enum" in schema:
                # Create enum for string literals
                enum_values = schema["enum"]
                return tb.union([tb.literal_string(val) for val in enum_values])
            return tb.string()
        elif schema_type == "number":
            return tb.float()
        elif schema_type == "integer":
            return tb.int()
        elif schema_type == "boolean":
            return tb.bool()
        elif schema_type == "array":
            item_type = self._get_baml_field_type(
                schema.get("items", {"type": "string"}), tb, required=True
            )
            return item_type.list()
        elif schema_type == "object":
            # Handle nested object by creating a new class
            if "properties" in schema:
                # Generate random name for nested class
                import random
                import string

                class_name = "".join(random.choices(string.ascii_lowercase, k=10))
                nested_class = tb.add_class(class_name)

                nested_required = schema.get("required", [])
                for prop_name, prop_schema in schema["properties"].items():
                    prop_required = prop_name in nested_required
                    prop_type = self._get_baml_field_type(
                        prop_schema, tb, prop_required
                    )

                    if not prop_required:
                        prop_type = prop_type.optional()

                    property_builder = nested_class.add_property(prop_name, prop_type)
                    if "description" in prop_schema:
                        property_builder.description(prop_schema["description"])

                return nested_class.type()
            else:
                return tb.map(tb.string(), tb.string())
        else:
            # Default to string for unknown types
            return tb.string()

    def _build_tool_types(
        self, tools: List[Dict[str, Any]], tb: TypeBuilder, test_category: str
    ) -> None:
        """Build BAML types from tool schemas and add to TypeBuilder."""
        if not tools:
            return

        is_simple = test_category in [
            "relevance",
            "java",
            "javascript",
            "simple",
            "parallel_function",
            "executable_simple",
            "executable_parallel_function",
            "rest",
        ]
        is_multiple = test_category in [
            "multiple_function",
            "parallel_multiple_function",
            "executable_multiple_function",
            "executable_parallel_multiple_function",
        ]

        if is_simple:
            # For simple functions, Response directly contains the tool parameters
            assert len(tools) == 1
            tool = tools[0]

            # Extract function info
            if "function" in tool:
                func_schema = tool["function"]["parameters"]
                func_name = tool["function"]["name"]
                func_description = tool["function"].get("description", "")
            else:
                func_schema = tool.get("parameters", {})
                func_name = tool.get("name", "UnknownFunction")
                func_description = tool.get("description", "")

            # Add function_name for relevance test category
            if "relevance" in test_category:
                # Create enum with function name
                func_enum = tb.add_enum(f"{func_name}_enum")
                func_enum.add_value(func_name)
                tb.Response.add_property("function_name", func_enum.type()).description(
                    func_description
                )

            # Add all function parameters to Response class
            required_props = func_schema.get("required", [])
            for prop_name, prop_schema in func_schema.get("properties", {}).items():
                prop_type = self._get_baml_field_type(prop_schema, tb)

                if prop_name not in required_props:
                    prop_type = prop_type.optional()

                property_builder = tb.Response.add_property(prop_name, prop_type)
                if "description" in prop_schema:
                    description = prop_schema["description"]
                    if "default" in prop_schema and prop_schema["default"]:
                        description += f". Default to '{prop_schema['default']}'"
                    property_builder.description(description)

        elif is_multiple:
            # For multiple functions, Response contains a union of function classes
            tool_classes = []

            for tool in tools:
                # Extract function info
                if "function" in tool:
                    func_schema = tool["function"]["parameters"]
                    func_name = tool["function"]["name"]
                    func_description = tool["function"].get("description", "")
                else:
                    func_schema = tool.get("parameters", {})
                    func_name = tool.get("name", "UnknownFunction")
                    func_description = tool.get("description", "")

                # Create class for this function
                func_class = tb.add_class(func_name)

                # Add function_name enum
                func_enum = tb.add_enum(f"{func_name}_enum")
                func_enum.add_value(func_name)
                func_class.add_property("function_name", func_enum.type()).description(
                    func_description
                )

                # Add all function parameters
                required_props = func_schema.get("required", [])
                for prop_name, prop_schema in func_schema.get("properties", {}).items():
                    prop_type = self._get_baml_field_type(prop_schema, tb)

                    if prop_name not in required_props:
                        prop_type = prop_type.optional()

                    property_builder = func_class.add_property(prop_name, prop_type)
                    if "description" in prop_schema:
                        description = prop_schema["description"]
                        if "default" in prop_schema and prop_schema["default"]:
                            description += f". Default to '{prop_schema['default']}'"
                        property_builder.description(description)

                tool_classes.append(func_class.type())

            # Add union property to Response
            tb.Response.add_property("function", tb.union(tool_classes))

    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for inclusion in the prompt."""
        tool_descriptions = []

        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                name = func["name"]
                description = func.get("description", "")
                parameters = func.get("parameters", {})
            else:
                name = tool.get("name", "UnknownFunction")
                description = tool.get("description", "")
                parameters = tool.get("parameters", {})

            tool_desc = f"{name}: {description}"

            # Add parameter information
            if "properties" in parameters:
                params = []
                required = parameters.get("required", [])
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "")
                    required_marker = " (required)" if param_name in required else ""
                    params.append(
                        f"  - {param_name} ({param_type}){required_marker}: {param_desc}"
                    )

                if params:
                    tool_desc += "\n" + "\n".join(params)

            tool_descriptions.append(tool_desc)

        return "\n\n".join(tool_descriptions)

    def decode_ast(self, result, language="Python"):
        """Decode AST from BAML response."""
        # BAML results are already in FC format, so we need to handle them like FC models
        if result is None:
            return []

        decoded_output = []

        if isinstance(result, list):
            for invoked_function in result:
                if invoked_function and isinstance(invoked_function, dict):
                    name = list(invoked_function.keys())[0]
                    params_json = invoked_function[name]

                    # Parse the JSON string back to dict (same as FC handler)
                    try:
                        params = (
                            json.loads(params_json)
                            if isinstance(params_json, str)
                            else params_json
                        )

                        # Convert values to strings for Java/JavaScript
                        if language in ["Java", "JavaScript"]:
                            for key in params:
                                params[key] = str(params[key])

                        decoded_output.append({name: params})
                    except (json.JSONDecodeError, TypeError):
                        # Handle case where params_json is not valid JSON
                        decoded_output.append({name: {}})

        return decoded_output

    def decode_execute(self, result):
        """Decode execution format from BAML response."""
        if result is None:
            return []

        execution_list = []

        # result should be a list of function calls from _parse_query_response_FC
        if isinstance(result, list):
            for invoked_function in result:
                if invoked_function and isinstance(invoked_function, dict):
                    name = list(invoked_function.keys())[0]
                    params_json = invoked_function[name]

                    # Parse the JSON string back to dict
                    try:
                        params = (
                            json.loads(params_json)
                            if isinstance(params_json, str)
                            else params_json
                        )
                        # Remove None values and function_name if it exists
                        params = {
                            key: value
                            for key, value in params.items()
                            if value is not None and key != "function_name"
                        }

                        # Format as function call
                        execution_list.append(
                            f"{name}({','.join([f'{k}={repr(v)}' for k, v in params.items()])})"
                        )
                    except (json.JSONDecodeError, TypeError):
                        # Handle case where params_json is not valid JSON
                        execution_list.append(f"{name}()")

        return execution_list

    #### FC methods ####

    def _determine_baml_function(self, test_entry: dict) -> str:
        """Determine which BAML function to call based on test characteristics."""
        functions = test_entry.get("function", [])
        test_id = test_entry.get("id", "")
        num_functions = len(functions)

        # Check test ID for hints about test type
        if "parallel" in test_id.lower():
            if num_functions == 1:
                return "ParallelFunction"
            else:
                return "ParallelMultipleFunctions"
        elif "multiple" in test_id.lower():
            return "MultipleFunctions"
        elif "relevance" in test_id.lower():
            return "RelevanceFunction"
        else:
            # Default to SimpleFunction for single function calls
            return "SimpleFunction"

    def _query_FC(self, inference_data: dict):
        """Execute BAML function call with retry logic."""
        user_query = inference_data["user_query"]
        functions_data = inference_data["functions_data"]
        client_registry = inference_data["client_registry"]
        type_builder = inference_data["type_builder"]
        baml_function_name = inference_data["baml_function_name"]
        test_category = inference_data["test_category"]

        # Log the input
        inference_data["inference_input_log"] = {
            "user_query": user_query,
            "functions_data": functions_data,
            "model": self.original_model_name,
            "baml_function": baml_function_name,
        }

        # Mock classes for responses
        class MockResponse:
            def __init__(self, result, test_category, functions_data):
                self.result = result
                self.test_category = test_category
                self.functions_data = functions_data
                self.choices = [MockChoice(result)]

        class MockChoice:
            def __init__(self, result):
                self.message = MockMessage(result)

        class MockMessage:
            def __init__(self, result):
                self.content = (
                    json.dumps(result) if isinstance(result, dict) else str(result)
                )

        class MockErrorResponse:
            def __init__(self, error):
                self.result = str(error)
                self.choices = [MockErrorChoice(str(error))]

        class MockErrorChoice:
            def __init__(self, result):
                self.message = MockErrorMessage(result)

        class MockErrorMessage:
            def __init__(self, result):
                self.content = (
                    json.dumps(result) if isinstance(result, dict) else str(result)
                )

        # Retry configuration
        max_retries = 3
        retry_delay = 1  # seconds

        start_time = time.time()
        last_exception = None

        for attempt in range(max_retries):
            try:
                # Get the appropriate BAML function
                baml_function = getattr(b, baml_function_name)

                # Call BAML function with Function[] format
                result = baml_function(
                    functions=functions_data,
                    query=user_query,
                    baml_options={
                        "client_registry": client_registry,
                        "tb": type_builder,
                    },
                )
                end_time = time.time()

                return (
                    MockResponse(result, test_category, functions_data),
                    end_time - start_time,
                )

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Final attempt failed
                    end_time = time.time()
                    return MockErrorResponse(e), end_time - start_time

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """Pre-process query for FC mode."""
        inference_data["user_query"] = ""  # Will be set later
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """Compile tools and create BAML types."""
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        # Process functions like other handlers
        functions = func_doc_language_specific_pre_processing(functions, test_category)

        # Apply the same transformations as FC handlers (including dot-to-underscore conversion)
        transformed_tools = convert_to_tool(
            functions, GORILLA_TO_OPENAPI, ModelStyle.OpenAI_Completions
        )

        # Determine which BAML function to call
        baml_function_name = self._determine_baml_function(test_entry)

        # Convert to BAML format
        tb = TypeBuilder()
        self._build_tool_types(functions, tb, test_category)

        # Create client registry
        client_registry = self._create_client_registry()

        # Convert functions to the format expected by BAML functions
        # Use the transformed tools to get the correct function names (with dots converted to underscores)
        functions_data = []
        for tool in transformed_tools:
            if "function" in tool:
                func_info = tool["function"]
                functions_data.append(
                    {
                        "name": func_info["name"],
                        "description": func_info.get("description", ""),
                    }
                )
            else:
                functions_data.append(
                    {
                        "name": tool.get("name", "UnknownFunction"),
                        "description": tool.get("description", ""),
                    }
                )

        inference_data["type_builder"] = tb
        inference_data["client_registry"] = client_registry
        inference_data["functions_data"] = functions_data
        inference_data["baml_function_name"] = baml_function_name
        inference_data["test_category"] = test_category
        inference_data["tools"] = functions  # Keep original for compatibility

        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """Parse BAML response."""
        try:
            # Extract result from BAML response
            result = api_response.result
            test_category = getattr(api_response, "test_category", "simple")
            functions_data = getattr(api_response, "functions_data", [])

            is_simple = test_category in [
                "relevance",
                "java",
                "javascript",
                "simple",
                "parallel_function",
                "executable_simple",
                "executable_parallel_function",
                "rest",
            ]
            is_multiple = test_category in [
                "multiple_function",
                "parallel_multiple_function",
                "executable_multiple_function",
                "executable_parallel_multiple_function",
            ]

            model_responses = []
            tool_call_ids = []

            # Handle case where BAML returns a Response object (Pydantic model)
            if hasattr(result, "__class__") and "Response" in str(type(result)):
                # BAML Response object - use Pydantic's model_dump() method for proper conversion
                try:
                    # Use Pydantic's model_dump() method to convert to dict
                    result_dict = result.model_dump()
                except (AttributeError, TypeError):
                    # Fallback: try to access result as dict-like object or __dict__
                    try:
                        result_dict = (
                            dict(result)
                            if hasattr(result, "__iter__") and hasattr(result, "items")
                            else result.__dict__
                        )
                    except (AttributeError, TypeError):
                        result_dict = {}
                        for attr in dir(result):
                            if not attr.startswith("_") and not callable(
                                getattr(result, attr, None)
                            ):
                                try:
                                    result_dict[attr] = getattr(result, attr)
                                except (AttributeError, TypeError):
                                    continue

                # For simple functions, infer function name from functions_data
                if is_simple and functions_data and len(functions_data) == 1:
                    func_name = functions_data[0]["name"]

                    # Filter out None values and function_name if it exists
                    params = {
                        k: v
                        for k, v in result_dict.items()
                        if v is not None and k != "function_name"
                    }

                    # Create the FC format response
                    model_responses = [{func_name: json.dumps(params)}]
                    tool_call_ids = [func_name]
                else:
                    # For other test types, we'll need to implement parsing logic
                    # For now, return empty response
                    model_responses = []
                    tool_call_ids = []

            elif isinstance(result, list):
                # Response[] - multiple function calls (parallel functions)
                for call in result:
                    if isinstance(call, dict):
                        if is_simple:
                            # For simple parallel functions, extract function_name if exists
                            func_name = call.get("function_name")
                            if func_name:
                                params = {
                                    k: v
                                    for k, v in call.items()
                                    if k != "function_name" and v is not None
                                }
                                model_responses.append({func_name: json.dumps(params)})
                                tool_call_ids.append(func_name)
                            else:
                                # Parallel function without function_name - infer from the single function
                                functions_data = getattr(
                                    api_response, "functions_data", []
                                )
                                if functions_data and len(functions_data) == 1:
                                    func_name = functions_data[0]["name"]
                                    params = {
                                        k: v for k, v in call.items() if v is not None
                                    }
                                    model_responses.append(
                                        {func_name: json.dumps(params)}
                                    )
                                    tool_call_ids.append(func_name)
                        elif is_multiple and "function" in call:
                            # For multiple parallel functions, handle the function union
                            func_data = call["function"]
                            func_name = func_data.get("function_name")
                            if func_name:
                                params = {
                                    k: v
                                    for k, v in func_data.items()
                                    if k != "function_name" and v is not None
                                }
                                model_responses.append({func_name: json.dumps(params)})
                                tool_call_ids.append(func_name)

            elif isinstance(result, dict):
                # Response - single function call
                if is_simple:
                    # For simple functions, check if function_name exists (relevance) or just use the parameters
                    func_name = result.get("function_name")
                    if func_name:
                        # Relevance function with function_name
                        params = {
                            k: v
                            for k, v in result.items()
                            if k != "function_name" and v is not None
                        }
                        model_responses = [{func_name: json.dumps(params)}]
                        tool_call_ids = [func_name]
                    else:
                        # Simple function without function_name - infer from the single function
                        functions_data = getattr(api_response, "functions_data", [])
                        if functions_data and len(functions_data) == 1:
                            func_name = functions_data[0]["name"]
                            params = {k: v for k, v in result.items() if v is not None}
                            model_responses = [{func_name: json.dumps(params)}]
                            tool_call_ids = [func_name]
                        else:
                            model_responses = []
                            tool_call_ids = []

                elif is_multiple and "function" in result:
                    # For multiple functions, extract from the function union
                    func_data = result["function"]
                    func_name = func_data.get("function_name")
                    if func_name:
                        params = {
                            k: v
                            for k, v in func_data.items()
                            if k != "function_name" and v is not None
                        }
                        model_responses = [{func_name: json.dumps(params)}]
                        tool_call_ids = [func_name]

            elif result is None:
                # Response? - optional response (RelevanceFunction)
                model_responses = []
                tool_call_ids = []

            else:
                # Fallback: treat as text response
                model_responses = str(result)
                tool_call_ids = []

        except Exception as e:
            # Handle error case
            model_responses = str(
                getattr(api_response, "choices", [{"message": {"content": str(e)}}])[0][
                    "message"
                ]["content"]
            )
            tool_call_ids = []

        # Mock message for chat history
        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.role = "assistant"
                self.tool_calls = None

        model_responses_message_for_chat_history = MockMessage(
            json.dumps(model_responses)
            if isinstance(model_responses, list)
            else model_responses
        )

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": 0,  # BAML doesn't provide token counts by default
            "output_token": 0,
            "reasoning_content": "",
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """Add first turn message for FC mode."""
        # Extract user query from message
        user_query = ""
        for msg in first_turn_message:
            if msg["role"] == "user":
                user_query = msg["content"]
                break

        inference_data["user_query"] = user_query
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """Add next turn user message for FC mode."""
        # For multi-turn, we would need to maintain conversation state
        # For now, just use the latest user message
        user_query = ""
        for msg in user_message:
            if msg["role"] == "user":
                user_query = msg["content"]
                break

        inference_data["user_query"] = user_query
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, assistant_message: dict
    ) -> dict:
        """Add assistant message for FC mode."""
        # For conversation history - would need to maintain state for multi-turn
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[dict],
        tool_call_ids: list[str],
    ) -> dict:
        """Add execution results for FC mode."""
        # For function execution results - would need to format and add to conversation
        return inference_data

    #### Prompting methods (not used with BAML but required by base class) ####

    def _query_prompting(self, inference_data: dict):
        """Not used with BAML."""
        raise NotImplementedError("BAML handler only supports FC mode")

    def _pre_query_processing_prompting(
        self, inference_data: dict, test_entry: dict
    ) -> dict:
        """Not used with BAML."""
        raise NotImplementedError("BAML handler only supports FC mode")

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """Not used with BAML."""
        raise NotImplementedError("BAML handler only supports FC mode")

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """Not used with BAML."""
        raise NotImplementedError("BAML handler only supports FC mode")

    def _add_assistant_message_prompting(
        self, inference_data: dict, assistant_message: dict
    ) -> dict:
        """Not used with BAML."""
        raise NotImplementedError("BAML handler only supports FC mode")

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[dict]
    ) -> dict:
        """Not used with BAML."""
        raise NotImplementedError("BAML handler only supports FC mode")
