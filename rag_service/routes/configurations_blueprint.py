import os
from flask import Blueprint, request, jsonify
from models import RagConfiguration
from utils import DB, LLM, Collection
from utils.validators import validate_configuration_request
from controller import Agent, initialize_settings, Index, QueryEngine
from environ import STORAGE

configurations_blueprint = Blueprint("configurations", __name__)


@configurations_blueprint.get("/configurations/")
def get_configurations():
    configurations = RagConfiguration.query.all()
    return jsonify([configuration_extract_data(c) for c in configurations]), 200


@configurations_blueprint.get("/configurations/<config_id>/")
def get_configuration_by_id(config_id):
    configuration = RagConfiguration.query.get_or_404(config_id)
    return jsonify(configuration_extract_data(configuration)), 200


@configurations_blueprint.post("/configurations/")
def post_configuration():
    try:
        configuration, warnings = validate_configuration_request(request)
        initialize_settings(configuration)
        documents = Collection().documents()
        vector_store_dir = os.path.join(STORAGE, f"{configuration.vector_store_name}")
        Index.from_documents(documents).persist(vector_store_dir)
        DB.session.add(configuration)
        DB.session.commit()

    except ValueError as e:
        return jsonify(message=str(e)), 400

    return jsonify(
        configuration=configuration_extract_data(configuration),
        warnings=warnings if warnings else None,
        ), 200


def configuration_extract_data(configuration: RagConfiguration) -> dict:
    return {
        "config_id": configuration.config_id,
        "system_prompt": configuration.system_prompt,
        "model_name": configuration.model_name,
        "embed_model_name": configuration.embed_model_name,
        "chunk_size": configuration.chunk_size,
        "chunk_overlap": configuration.chunk_overlap,
        "temperature": configuration.temperature,
        "top_k": configuration.top_k,
        "top_p": configuration.top_p,
    }