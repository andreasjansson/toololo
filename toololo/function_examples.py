EXAMPLES = {}


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of the two numbers
    """
    return a + b


EXAMPLES[add] = {
    "name": "add",
    "description": "Add two numbers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "a": {"description": "First number", "type": "integer"},
            "b": {"description": "Second number", "type": "integer"},
        },
        "required": ["a", "b"],
    },
}


def process_data(
    items: list[int], config: dict = None, threshold: float = 0.5
) -> tuple[int, list[str], dict]:
    """Process a list of integers with optional configuration.

    This function takes a list of integers and processes them according to the
    provided configuration. If no configuration is provided, default settings
    are used. The threshold controls the sensitivity of the processing algorithm.

    Args:
        items: List of integers to process. Each integer should be positive.
        config: Optional configuration dictionary with processing parameters.
            Can contain keys like 'mode', 'precision', etc.
        threshold: Sensitivity threshold for the processing algorithm.
            Values range from 0.0 to 1.0, with higher values being more strict.

    Returns:
        A tuple containing:
        - Count of processed items (int)
        - List of processed items as strings (list[str])
        - Statistics about the processing (dict)

    Raises:
        ValueError: If threshold is not between 0 and 1
        TypeError: If items contains non-integer values
    """
    return len(items), [str(i) for i in items], {"avg": sum(items) / len(items)}


EXAMPLES[process_data] = {
    "name": "process_data",
    "description": "Process a list of integers with optional configuration. This function takes a list of integers and processes them according to the provided configuration. If no configuration is provided, default settings are used. The threshold controls the sensitivity of the processing algorithm.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "description": "List of integers to process. Each integer should be positive.",
                "type": "array",
                "items": {"type": "integer"},
            },
            "config": {
                "description": "Optional configuration dictionary with processing parameters. Can contain keys like 'mode', 'precision', etc.",
                "type": "object",
            },
            "threshold": {
                "description": "Sensitivity threshold for the processing algorithm. Values range from 0.0 to 1.0, with higher values being more strict.",
                "type": "number",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["items"],
    },
    "errors": {
        "ValueError": "If threshold is not between 0 and 1",
        "TypeError": "If items contains non-integer values",
    },
}


def analyze_input(
    value: int | str | None, options: list[dict] = None, debug: bool = False
) -> dict[str, any]:
    """Analyze a value that can be an integer, string, or None.

    This function performs complex analysis on the input value
    based on its type and content. The analysis can be customized
    using the options parameter.

    Args:
        value: The value to analyze. Can be an integer (for numerical analysis),
              a string (for text analysis), or None (for default analysis).
        options: List of option dictionaries to configure the analysis.
                Each dict should have 'name' and 'value' keys.
        debug: When True, additional debug information is included in the result.

    Returns:
        A dictionary containing the analysis results with the following keys:
        - 'type': The detected type of the input
        - 'length': Size or length of the input (when applicable)
        - 'summary': Text summary of the analysis
        - 'details': Detailed analysis results
        - 'debug_info': Debug information (only when debug=True)
    """
    return {"type": "unknown"}


EXAMPLES[analyze_input] = {
    "name": "analyze_input",
    "description": "Analyze a value that can be an integer, string, or None. This function performs complex analysis on the input value based on its type and content. The analysis can be customized using the options parameter.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {
                "description": "The value to analyze. Can be an integer (for numerical analysis), a string (for text analysis), or None (for default analysis).",
                "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}],
            },
            "options": {
                "description": "List of option dictionaries to configure the analysis. Each dict should have 'name' and 'value' keys.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "object"},
                    },
                    "required": ["name", "value"],
                },
            },
            "debug": {
                "description": "When True, additional debug information is included in the result.",
                "type": "boolean",
                "default": False,
            },
        },
        "required": ["value"],
    },
}
