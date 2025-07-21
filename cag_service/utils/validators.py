from flask import Request


def validate_configuration_request(request: Request) -> tuple[str, str, list[str]]:
    data = _validate_data(request)

    system_prompt = data.get("system_prompt")
    model_name = data.get("model_name")

    errors: list[str] = []
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        errors.append("No system prompt provided")

    if not isinstance(model_name, str) or not model_name.strip():
        errors.append("No model name provided")

    if len(errors) > 0:
        print(errors)
        raise ValueError("Invalid request: " + ", ".join(errors))

    return str(system_prompt), str(model_name), errors


def validate_cag_chat_request(request: Request) -> str:
    data = _validate_data(request)

    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Invalid request: No query provided")

    return query


def _validate_data(request: Request):
    data = request.get_json() if request.is_json else request.form
    if not data:
        raise ValueError("Invalid request: No data provided")
    return data
