# BAML Integration Plan for Berkeley Function Call Leaderboard

## Overview
This document outlines the plan to add BAML (Boundary ML) support to the Berkeley Function Call Leaderboard (BFCL). The integration will allow running all LLM calls through BAML's infrastructure using dynamic types and the TypeBuilder API.

## Goals
1. Add a new CLI flag `--tool-format baml` to enable BAML mode
2. Convert tool JSON schemas to BAML dynamic types at runtime
3. Route ALL LLM calls through a single BAML handler when flag is set
4. Auto-detect provider from model name and base_url
5. Transform multi-tool calls into appropriate BAML structures (unions or classes)
6. Minimize code changes for long-term maintainability

## Implementation Plan

### Phase 1: CLI Integration (Minimal Changes)
1. **Add CLI Flag** (`bfcl_eval/__main__.py`)
   - Add `--tool-format` parameter with options: `default`, `baml`
   - Pass this flag through the generation pipeline
   - Default to existing behavior when not specified
   - No need to modify ModelConfig or add new fields

### Phase 2: BAML Infrastructure Setup
1. **Create BAML Utilities Module** (`bfcl_eval/model_handler/baml_utils.py`)
   - Import the provided `parse_json_schema` function
   - Create helper functions for TypeBuilder operations
   - Implement tool schema to BAML type conversion

2. **Define Base BAML Types** (`baml_config/`)
   - Create a `.baml` file with base function definition
   - Define a dynamic output type that will be modified at runtime
   - Example:
     ```baml
     class ToolOutput {
       @@dynamic
     }
     
     function CallTools(prompt: string, tools: string) -> ToolOutput {
       client DynamicClient
       prompt #"
         {{ prompt }}
         
         Available tools:
         {{ tools }}
         
         {{ ctx.output_format }}
       "#
     }
     ```

### Phase 3: BAML Handler Implementation
1. **Create Universal BAML Handler** (`bfcl_eval/model_handler/baml_handler.py`)
   ```python
   class BAMLHandler(BaseHandler):
       def __init__(self, model_name: str, temperature: float = 0.0):
           super().__init__(model_name, temperature)
           # Extract base_url if needed from existing handlers
           self.base_url = self._get_base_url_from_config()
       
       def _get_base_url_from_config(self) -> Optional[str]:
           """Extract base_url from existing model config if available."""
           # Check if model has a known base_url in existing handlers
           # This allows reusing existing configuration
           return None  # Will be auto-detected if not found
   ```

2. **Key Methods**:
   - `_build_tool_types()`: Convert JSON schemas to BAML types
   - `_create_client()`: Auto-detect provider and create client
   - `_format_prompt()`: Prepare prompts for BAML function calls
   - `_parse_baml_response()`: Convert BAML responses to expected format

### Phase 4: Tool Schema Conversion
1. **Schema Parser Integration** (`bfcl_eval/model_handler/baml_utils.py`)
   - Use provided `parse_json_schema` function
   - Handle special cases:
     - Multi-tool calls → Union types
     - Single tool calls → Direct class types
     - Nested parameters → Nested BAML classes
   - Add tool name tracking using `TOOL_NAME_KEY`

2. **Dynamic Type Building Process**:
   ```python
   def build_baml_types(tools, tb: TypeBuilder):
       if len(tools) == 1:
           # Single tool: create a class
           tool_type = parse_json_schema(tools[0]['parameters'], tb)
       else:
           # Multiple tools: create a union
           tool_types = []
           for tool in tools:
               schema = tool['parameters']
               schema['properties'][TOOL_NAME_KEY] = {
                   'type': 'string',
                   'enum': [tool['name']]
               }
               tool_type = parse_json_schema(schema, tb)
               tool_types.append(tool_type)
           tool_type = tb.union(tool_types)
       
       tb.ToolOutput.add_property('result', tool_type)
   ```

### Phase 5: Simple Provider Auto-Detection

1. **Provider Detection** (`bfcl_eval/model_handler/baml_handler.py`)
   ```python
   def detect_baml_provider(model_name: str, base_url: Optional[str] = None) -> tuple[str, dict]:
       """Auto-detect BAML provider and return (provider, options)."""
       model_lower = model_name.lower()
       
       # Simple prefix detection
       if model_lower.startswith(('gpt-', 'o1-')):
           return 'openai', {}
       elif model_lower.startswith('o3-'):
           return 'openai-responses', {}
       elif model_lower.startswith('claude-'):
           return 'anthropic', {}
       elif model_lower.startswith(('gemini-', 'models/gemini-')):
           return 'google-ai', {}
       elif model_lower.startswith('bedrock/'):
           return 'aws-bedrock', {}
       elif model_lower.startswith('azure/'):
           return 'azure-openai', {}
       else:
           # Default to openai-generic with base_url
           if not base_url:
               # Try to infer base_url from known providers
               if 'llama' in model_lower:
                   base_url = 'https://llama-api.meta.com/compat/v1'
               elif 'mistral' in model_lower:
                   base_url = 'https://api.mistral.ai/v1'
               elif 'deepseek' in model_lower:
                   base_url = 'https://api.deepseek.com/v1'
               elif 'qwen' in model_lower:
                   base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
               else:
                   raise ValueError(
                       f"Cannot auto-detect provider for model '{model_name}'. "
                       f"BAML supports: OpenAI, Anthropic, Google AI, AWS Bedrock, "
                       f"Azure OpenAI, and OpenAI-compatible APIs."
                   )
           
           return 'openai-generic', {'base_url': base_url}
   ```

2. **Simple Client Creation** (`bfcl_eval/model_handler/baml_handler.py`)
   ```python
   def create_baml_client(model_name: str, api_key: str, base_url: Optional[str] = None, **kwargs):
       """Create BAML client with auto-detected provider."""
       from baml_py import ClientRegistry
       
       cr = ClientRegistry()
       provider, provider_options = detect_baml_provider(model_name, base_url)
       
       # Merge provider options with common options
       options = {
           'model': model_name,
           'api_key': api_key,
           **provider_options,
           **kwargs  # Allow override of any options
       }
       
       cr.add_llm_client(
           name='DynamicClient',
           provider=provider,
           options=options
       )
       cr.set_primary('DynamicClient')
       return cr
   ```

### Phase 6: Integration Points (Minimal Changes)
1. **Single Integration Point** (`_llm_response_generation.py`)
   ```python
   # In the generate() function, add this check:
   if tool_format == "baml":
       # Route ALL models through BAML handler
       handler = BAMLHandler(model_name=model, temperature=temperature)
       # Continue with normal flow
   else:
       # Use existing handler selection logic
       handler = get_handler_for_model(model)
   ```

2. **No Model Configuration Changes**
   - No need to modify ModelConfig
   - No need to add supports_baml flags
   - BAML handler will handle all models and fail gracefully for unsupported ones

### Phase 7: Testing and Validation
1. **Unit Tests**
   - Test JSON schema to BAML type conversion
   - Verify provider mapping
   - Test single and multi-tool scenarios

2. **Integration Tests**
   - Run subset of BFCL tests with BAML flag
   - Compare results with standard implementation
   - Ensure consistent output format

## Implementation Order
1. Set up BAML dependencies and utilities
2. Create basic BAML handler with single-tool support
3. Implement multi-tool union support
4. Add provider mapping and client registry
5. Integrate with CLI and test
6. Extend to all supported providers

## Key Considerations
- **Error Handling**: Gracefully handle unsupported providers
- **Performance**: Cache TypeBuilder instances when possible
- **Compatibility**: Ensure output format matches existing BFCL expectations
- **Logging**: Add detailed logging for debugging BAML operations
- **Documentation**: Update README with BAML usage instructions

## File Structure
```
bfcl_eval/
├── model_handler/
│   ├── baml_handler.py         # New: BAML handler implementation
│   ├── baml_utils.py           # New: BAML utilities and schema parser
│   └── utils.py                # Modified: Add BAML tool conversion
├── baml_config/                # New: BAML configuration files
│   └── tools.baml              # Base BAML definitions
└── __main__.py                 # Modified: Add --tool-format flag
```

## Example Usage

### Running with BAML
```bash
# OpenAI models (auto-detected)
bfcl generate --model gpt-4o --tool-format baml

# Anthropic models (auto-detected)
bfcl generate --model claude-3-5-sonnet --tool-format baml

# Models with custom base URLs (auto-detected from model name)
bfcl generate --model llama-3.3-70b --tool-format baml

# Multiple models at once
bfcl generate --model gpt-4o,claude-3-5-sonnet,gemini-2.0-flash --tool-format baml

# With existing flags
bfcl generate --model gpt-4o --tool-format baml --test-category simple --temperature 0.7
```

### How It Works
1. User adds `--tool-format baml` to any generate command
2. ALL models specified are routed through BAMLHandler
3. BAMLHandler auto-detects the provider based on model name
4. For unrecognized models, it defaults to openai-generic with appropriate base_url
5. If provider is not supported by BAML, it throws a clear error message

## Success Criteria
- Users can run `bfcl generate --model gpt-4 --tool-format baml`
- All BAML-supported providers work correctly
- Output format remains compatible with existing evaluation pipeline
- Performance is comparable to direct API calls
- Clear error messages for unsupported features
- Minimal code changes (only add BAMLHandler and one CLI flag)