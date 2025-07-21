import torch
from flask import Blueprint, request, jsonify
from controller import initialize_cache, get_answer, clean_up_cache
from models import CagConfiguration
from utils.validators import validate_cag_chat_request
from utils import LLM


cag_blueprint = Blueprint("cag", __name__)


@cag_blueprint.post("/<config_id>/chat")
def cag_chat(config_id):
    config: CagConfiguration = CagConfiguration.query.get_or_404(config_id)
    try:
        llm = LLM(config.model_name)
        query = validate_cag_chat_request(request)
        cache_path = initialize_cache(config, llm)
        cache = torch.load(cache_path, weights_only=False)
        answer = get_answer(query, llm.tokenizer(), llm.model(), llm.device(), cache)
        print(f"Answer: {answer}")
        clean_up_cache(cache)
    except ValueError as e:
        return jsonify(message=str(e)), 400

    return jsonify(answer=answer), 200
