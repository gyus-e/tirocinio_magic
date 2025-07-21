from flask import Request
from models import RagConfiguration

def validate_configuration_request(request: Request) -> tuple[RagConfiguration, list[str]]:
    data = _validate_data(request)

    system_prompt = data.get("system_prompt")
    model_name = data.get("model_name")
    embed_model_name = data.get("embed_model_name")
    chunk_size = data.get("chunk_size", 512)
    chunk_overlap = data.get("chunk_overlap", 64)
    temperature = data.get("temperature", 0.4)
    top_k = data.get("top_k", None)
    top_p = data.get("top_p", None)

    errors: list[str] = []
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        errors.append("No system prompt provided")

    if not isinstance(model_name, str) or not model_name.strip():
        errors.append("No model name provided")
    
    if not isinstance(embed_model_name, str) or not embed_model_name.strip():
        errors.append("No embed model name provided")

    if len(errors) > 0:
        print(errors)
        raise ValueError("Invalid request: " + ", ".join(errors))

    return RagConfiguration(
        system_prompt=system_prompt,
        model_name=model_name,
        embed_model_name=embed_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    ), errors


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
