import inspect
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Any, TypeVar
from functools import wraps
import anthropic

from .function_examples import EXAMPLES


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


def function_to_jsonschema(
    client: anthropic.Client,
    model: str,
    func: Callable[..., Any],
    max_attempts: int = 5,
) -> dict:
    cache_dir = Path.home() / ".cache" / "toololo" / "schemas"
    cache_dir.mkdir(parents=True, exist_ok=True)

    hashed_name = hashed_function_name(func)
    cache_file = cache_dir / f"{hashed_name}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                schema = json.load(f)
            if validate_schema(schema):
                return schema
        except json.JSONDecodeError:
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
- "name": (will be filled in automatically)
- "description": a detailed description of what the function does
- "input_schema": a JSON Schema object describing the inputs

The input_schema MUST ONLY include these fields:
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
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        schema_text = response.content[0].text

        if "```json" in schema_text:
            schema_text = schema_text.split("```json")[1].split("```")[0]
        elif "```" in schema_text:
            schema_text = schema_text.split("```")[1].split("```")[0]

        try:
            schema = json.loads(schema_text.strip())
            schema["name"] = hashed_name
            if validate_schema(schema):
                # Save to cache
                with open(cache_file, "w") as f:
                    json.dump(schema, f, indent=2)
                return schema
            else:
                schema = None
        except (json.JSONDecodeError, ValueError) as e:
            schema = None

    raise ValueError(
        f"Failed to generate valid tool use schema after {max_attempts} attempts"
    )


def validate_schema(schema: dict) -> bool:
    """Validate the generated tool use schema for completeness and correctness."""
    try:
        # Check required top-level fields
        required_fields = ["name", "description", "input_schema"]
        allowed_fields = set(required_fields)

        # Check for extra top-level fields
        extra_fields = [field for field in schema if field not in allowed_fields]
        if extra_fields:
            return False

        # Validate name
        if not isinstance(schema["name"], str) or not schema["name"]:
            return False

        # Validate description
        if not isinstance(schema["description"], str) or not schema["description"]:
            return False

        # Check input_schema structure
        input_schema = schema.get("input_schema", {})
        if not isinstance(input_schema, dict):
            return False

        if input_schema.get("type") != "object":
            return False

        # Check for extra fields in input_schema
        allowed_input_schema_fields = {"type", "properties", "required"}
        extra_input_fields = [
            field for field in input_schema if field not in allowed_input_schema_fields
        ]
        if extra_input_fields:
            return False

        # Check properties exist and are correctly structured
        properties = input_schema.get("properties", None)
        if not isinstance(properties, dict) or properties is None:
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
        required = input_schema.get("required", [])
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


def make_compatible(func: Callable[..., T]) -> Callable[..., T]:
    """
    Wrap functions that don't support keyword arguments to make them compatible with
    the toololo framework.

    For functions that already accept kwargs, this returns the original function.
    For functions that don't accept kwargs, it creates a wrapper that unpacks kwargs
    into positional arguments based on parameter names.
    """
    sig = inspect.signature(func)
    accepts_kwargs = any(
        p.kind in (p.VAR_KEYWORD, p.KEYWORD_ONLY) for p in sig.parameters.values()
    )

    # If function already supports kwargs or has no parameters, return as-is
    if accepts_kwargs or not sig.parameters:
        return func

    @wraps(func)
    def wrapper(**kwargs):
        # Map keyword arguments to positional arguments based on parameter names
        param_names = list(sig.parameters.keys())
        args = []

        # Check if all provided kwargs exist in the function signature
        unknown_args = set(kwargs.keys()) - set(param_names)
        if unknown_args:
            raise TypeError(
                f"Got unexpected keyword arguments: {', '.join(unknown_args)}"
            )

        # Build positional args in the order defined in the function signature
        for name in param_names:
            if name in kwargs:
                args.append(kwargs[name])
            elif sig.parameters[name].default is not inspect.Parameter.empty:
                # Skip if parameter has a default value and no value provided
                continue
            else:
                raise TypeError(f"Missing required positional argument: '{name}'")

        return func(*args)

    return wrapper
