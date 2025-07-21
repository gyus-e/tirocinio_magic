from flask import Blueprint, request, jsonify
from models import CagConfiguration
from utils import DB, LLM
from utils.validators import validate_configuration_request
from controller import initialize_cache


configurations_blueprint = Blueprint("configurations", __name__)


@configurations_blueprint.get("/configurations/")
def get_configurations():
    configurations = CagConfiguration.query.all()
    return jsonify([configuration_extract_data(c) for c in configurations]), 200


@configurations_blueprint.get("/configurations/<config_id>/")
def get_configuration_by_id(config_id):
    configuration = CagConfiguration.query.get_or_404(config_id)
    return jsonify(configuration_extract_data(configuration)), 200


@configurations_blueprint.post("/configurations/")
def post_configuration():
    try:
        system_prompt, model_name, warnings = validate_configuration_request(request)
        configuration = CagConfiguration(
            system_prompt=system_prompt, 
            model_name=model_name,
            )
        llm = LLM(configuration.model_name)
        cache_path = initialize_cache(configuration, llm) #TODO: use async message queue
        DB.session.add(configuration)
        DB.session.commit()
    except ValueError as e:
        return jsonify(message=str(e)), 400

    return jsonify(
        configuration=configuration_extract_data(configuration),
        warnings=warnings if warnings else None,
        ), 200


def configuration_extract_data(configuration: CagConfiguration) -> dict:
    return {
        "config_id": configuration.config_id,
        "system_prompt": configuration.system_prompt,
        "model_name": configuration.model_name,
    }