
"""Validate tool arguments before any tool is executed.

The LLM proposes arguments, but Python decides whether
those arguments are acceptable.
"""


class ValidationError(ValueError):
    """Raised when tool arguments are invalid."""


def validate_arguments(parameters: dict, arguments) -> dict:

    # Arguments must be a JSON object.
    if not isinstance(arguments, dict):
        raise ValidationError(
            "Arguments must be a JSON object."
        )

    # Reject arguments that are not defined by the tool.
    unknown = set(arguments) - set(parameters)

    if unknown:
        raise ValidationError(
            f"Unexpected arguments: {sorted(unknown)}"
        )

    clean = {}

    # Validate each parameter defined by the tool.
    for name, rule in parameters.items():

        # Handle missing arguments.
        if name not in arguments:

            if rule.get("required"):
                raise ValidationError(
                    f"Missing required argument: {name}"
                )

            if "default" in rule:
                clean[name] = rule["default"]

            continue

        value = arguments[name]
        kind = rule["type"]

        # String validation.
        if kind == "string":

            if not isinstance(value, str):
                raise ValidationError(
                    f"{name} must be a string."
                )

            value = value.strip()

            if rule.get("required") and not value:
                raise ValidationError(
                    f"{name} cannot be empty."
                )

            if len(value) > rule.get("max_length", 1000):
                raise ValidationError(
                    f"{name} is too long."
                )

        # Integer validation.
        elif kind == "integer":

            # Allow numeric strings such as "3".
            if isinstance(value, str) and value.strip().isdigit():
                value = int(value.strip())

            # Booleans are also integers in Python,
            # so explicitly reject them.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(
                    f"{name} must be an integer."
                )

            minimum = rule.get("min")
            maximum = rule.get("max")

            if minimum is not None and value < minimum:
                raise ValidationError(
                    f"{name} must be at least {minimum}."
                )

            if maximum is not None and value > maximum:
                raise ValidationError(
                    f"{name} must not exceed {maximum}."
                )

        else:
            raise ValidationError(
                f"Unsupported argument type: {kind}"
            )

        clean[name] = value

    return clean
