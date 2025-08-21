import inspect
import hashlib
import json
import logging
from pathlib import Path
import re
import traceback
from typing import Callable, TypeVar, Awaitable, Union, Any, cast
from functools import wraps
import openai

from .function_examples import EXAMPLES

logger = logging.getLogger(__name__)


def compute_function_hash(func: Callable[..., Any]) -> str:
    # Add the source of function.py itself to the hash calculation
    function_py_path = Path(__file__)
    if function_py_path.exists():
        with open(function_py_path, "rb") as f:
            function_py_content = f.read()
    else:
        function_py_content = b""

    try:
        source = inspect.getsource(func)
        combined_source = source.encode() + function_py_content
        return hashlib.md5(combined_source).hexdigest()
    except (TypeError, OSError):
        # For built-in functions, use name, module, and signature
        module = func.__module__ if hasattr(func, "__module__") else "builtin"
        name = func.__name__ if hasattr(func, "__name__") else str(func)
        sig_str = str(inspect.signature(func)) if callable(func) else ""
        content = f"{module}.{name}{sig_str}"
        combined_content = content.encode() + function_py_content
        return hashlib.md5(combined_content).hexdigest()


def hashed_function_name(func: Callable[..., Any]) -> str:
    max_length = 64
    hash_length = 8
    hash_prefix = compute_function_hash(func)[:hash_length]
    sep = "-"
    sep_length = len(sep)

    function_name = func.__name__ if hasattr(func, "__name__") else "builtin_func"
    function_name = re.sub(r"[^a-zA-Z0-9_-]", "", function_name)

    if len(function_name) > max_length - hash_length - sep_length:
        function_name = function_name[: max_length - hash_length - sep_length]

    function_name = f"{function_name}{sep}{hash_prefix}"

    return function_name


def get_function_info(func: Callable[..., Any]) -> str:
    """Extract function information from signature and docstring when source code isn't available."""
    info = []

    try:
        # Get module and name
        module = func.__module__ if hasattr(func, "__module__") else "builtin"
        name = func.__name__ if hasattr(func, "__name__") else str(func)
        info.append(f"Function: {module}.{name}")

        # Get signature
        try:
            signature = inspect.signature(func)
            info.append(f"Signature: {signature}")

            # Get parameter details
            info.append("Parameters:")
            for param_name, param in signature.parameters.items():
                param_info = f"  - {param_name}"
                if param.annotation != inspect.Parameter.empty:
                    param_info += f" ({param.annotation})"
                if param.default != inspect.Parameter.empty:
                    param_info += f" = {param.default}"
                info.append(param_info)
        except (ValueError, TypeError):
            info.append("Signature: Unable to retrieve")

        # Get docstring
        if func.__doc__:
            info.append(f"Docstring:\n{func.__doc__.strip()}")
        else:
            info.append("Docstring: None")

        # Get return annotation if available
        try:
            if signature.return_annotation != inspect.Signature.empty:
                info.append(f"Return type: {signature.return_annotation}")
        except (ValueError, TypeError, UnboundLocalError):
            pass

    except Exception as e:
        info.append(f"Error extracting function info: {str(e)}")

    return "\n".join(info)


async def function_to_jsonschema(
    client: openai.AsyncOpenAI,
    model: str,
    func: Callable[..., Any],
    max_attempts: int = 5,
) -> dict:
    func_name = func.__name__ if hasattr(func, '__name__') else 'unknown'
    logger.debug(f"Generating schema for function {func_name}")
    
    cache_dir = Path.home() / ".cache" / "toololo" / "schemas"
    cache_dir.mkdir(parents=True, exist_ok=True)

    hashed_name = hashed_function_name(func)
    cache_file = cache_dir / f"{hashed_name}.json"

    if cache_file.exists():
        logger.debug(f"Checking cache for {func_name}")
        try:
            with open(cache_file, "r") as f:
                schema = json.load(f)
            if validate_schema(schema):
                logger.info(f"Using cached schema for {func_name}")
                return schema
            else:
                logger.warning(f"Cached schema for {func_name} is invalid, regenerating")
        except json.JSONDecodeError as e:
            logger.warning(f"Cache file for {func_name} is corrupted: {e}")
            # Ignore cache if it's invalid
            pass

    system_prompt = """You're an expert at converting Python functions to LLM tool use schemas. Analyze the provided function and create a detailed tool use schema that includes:
1. Parameters with their types and descriptions
2. Return values with types and descriptions
3. Required parameters based on default values
4. Proper handling of complex types, unions, etc.

IMPORTANT: Be thorough in your analysis. Extract all information from function signatures, type hints, and docstrings.
Create an exhaustive description that captures the full purpose and behavior of the function.

The valid tool schema MUST ONLY include these top-level fields:
- "type": must be "function" 
- "function": object with "name", "description", and "parameters"

The function object MUST ONLY include these fields:
- "name": (will be filled in automatically)
- "description": a detailed description of what the function does
- "parameters": a JSON Schema object describing the inputs

The parameters schema MUST ONLY include these fields:
- "type": must be "object"
- "properties": map of parameter names to their schemas
- "required": array of required parameter names

Property schemas MUST ONLY include these fields as appropriate:
- "type": standard JSON Schema type
- "description": detailed description of the parameter
- "enum": for enumerated values
- "items": for array types
- "anyOf" or "oneOf": for union types

Here are some examples of valid conversions:
"""

    # Add examples to the system message
    for i, (example_func, example_schema) in enumerate(EXAMPLES.items(), 1):
        example_source = inspect.getsource(example_func)
        system_prompt += f"\nExample {i}:\n```python\n{example_source}\n```\n\n"
        system_prompt += (
            f"Tool use schema:\n```json\n{json.dumps(example_schema, indent=2)}\n```\n"
        )

    system_prompt += """

Note that only input_schema is specified, so if you want to describe the output, that needs to go in the top-level description.

Only respond with the tool use schema in a JSON format, nothing else. Follow the examples above for structure and format. Extract as much detail as possible from the function's docstring, signature, and type hints to create a comprehensive schema.
"""

    # Try to get the source code, or fall back to function info
    try:
        source = inspect.getsource(func)
        user_message = (
            f"Create a tool use schema for this function:\n\n```python\n{source}\n```"
        )
    except (TypeError, OSError):
        # For built-in functions, use the signature and docstring
        function_info = get_function_info(func)
        user_message = f"Create a tool use schema for this function based on its information:\n\n{function_info}"

    schema = None
    attempt = 0

    while attempt < max_attempts and not schema:
        attempt += 1
        logger.info(f"Schema generation attempt {attempt}/{max_attempts} for {func_name}")
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
            )
            logger.debug(f"Schema generation API call successful for {func_name}")

            schema_text = response.choices[0].message.content
            logger.debug(f"Raw schema response for {func_name}: {schema_text[:200]}...")

            if "```json" in schema_text:
                schema_text = schema_text.split("```json")[1].split("```")[0]
            elif "```" in schema_text:
                schema_text = schema_text.split("```")[1].split("```")[0]

            try:
                schema = json.loads(schema_text.strip())
                # Set the function name in OpenAI format
                if "function" in schema and isinstance(schema["function"], dict):
                    schema["function"]["name"] = hashed_name
                if validate_schema(schema):
                    logger.info(f"Successfully generated valid schema for {func_name}")
                    # Save to cache
                    with open(cache_file, "w") as f:
                        json.dump(schema, f, indent=2)
                    logger.debug(f"Cached schema for {func_name}")
                    return schema
                else:
                    logger.warning(f"Generated schema for {func_name} failed validation: {schema}")
                    schema = None
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse schema JSON for {func_name}: {e}")
                logger.debug(f"Problematic JSON: {schema_text}")
                schema = None
                
        except Exception as e:
            logger.error(f"Error during schema generation API call for {func_name}: {e}")
            logger.error(f"Schema generation error: {traceback.format_exc()}")
            # Continue to next attempt

    raise ValueError(
        f"Failed to generate valid tool use schema after {max_attempts} attempts"
    )


def validate_schema(schema: dict) -> bool:
    """Validate the generated tool schema for OpenAI format completeness and correctness."""
    try:
        # Check required top-level fields for OpenAI format
        if schema.get("type") != "function":
            return False

        function_def = schema.get("function", {})
        if not isinstance(function_def, dict):
            return False

        # Check required function fields
        required_function_fields = ["name", "description", "parameters"]
        for field in required_function_fields:
            if field not in function_def:
                return False

        # Validate name
        if not isinstance(function_def["name"], str) or not function_def["name"]:
            return False

        # Validate description
        if not isinstance(function_def["description"], str) or not function_def["description"]:
            return False

        # Check parameters structure
        parameters = function_def.get("parameters", {})
        if not isinstance(parameters, dict):
            return False

        if parameters.get("type") != "object":
            return False

        # Check properties exist and are correctly structured
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            return False

        # Check each property has at least a description and type/anyOf/oneOf
        for prop_name, prop_info in properties.items():
            if not isinstance(prop_info, dict):
                return False

            if "description" not in prop_info:
                return False

            # Each property should have either type or anyOf/oneOf
            has_type = "type" in prop_info
            has_type_alternative = any(alt in prop_info for alt in ["anyOf", "oneOf"])

            if not (has_type or has_type_alternative):
                return False

        # Validate required field if present
        required = parameters.get("required", [])
        if not isinstance(required, list):
            return False

        # All required fields should exist in properties
        for field in required:
            if field not in properties:
                return False

        return True

    except Exception as e:
        return False


T = TypeVar("T")


def make_compatible(
    func: Union[Callable[..., T], Callable[..., Awaitable[T]]],
) -> Union[Callable[..., T], Callable[..., Awaitable[T]]]:
    """
    Wrap functions that don't support keyword arguments to make them compatible with
    the toololo framework.

    This function addresses three cases:
    1. Functions that already accept kwargs - return as-is
    2. Functions with only positional-or-keyword parameters - return as-is
    3. Functions with positional-only parameters - wrap to convert kwargs to args

    Works with both sync and async functions.
    """
    sig = inspect.signature(func)

    # If no parameters or doesn't need wrapping, return as-is
    if not sig.parameters:
        return func

    # Check if the function needs wrapping (has any POSITIONAL_ONLY params)
    needs_wrapping = any(
        p.kind == inspect.Parameter.POSITIONAL_ONLY for p in sig.parameters.values()
    )

    # If no wrapping needed, return as-is
    if not needs_wrapping:
        return func

    # Get ordered parameter names for mapping kwargs to args
    param_names = list(sig.parameters.keys())

    # Common logic to prepare positional arguments from kwargs
    def prepare_args(kwargs):
        # Check for unknown arguments
        unknown_args = set(kwargs.keys()) - set(param_names)
        if unknown_args:
            raise TypeError(
                f"Got unexpected keyword arguments: {', '.join(unknown_args)}"
            )

        # Map kwargs to positional args preserving order from signature
        args = []
        for name in param_names:
            if name in kwargs:
                args.append(kwargs[name])
            elif sig.parameters[name].default is not inspect.Parameter.empty:
                # Skip if parameter has default value and not provided
                continue
            else:
                raise TypeError(f"Missing required argument: '{name}'")

        return args

    # Create appropriate wrapper based on whether function is async or not
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(**kwargs):
            args = prepare_args(kwargs)
            return await func(*args)

        # Using cast to make type checkers happy about the return type
        return cast(Callable[..., Awaitable[T]], async_wrapper)
    else:

        @wraps(func)
        def wrapper(**kwargs):
            args = prepare_args(kwargs)
            return func(*args)

        # Using cast to make type checkers happy about the return type
        return cast(Callable[..., T], wrapper)
