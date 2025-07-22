import os
from flask import Blueprint, request, jsonify
from models import RagConfiguration
from controller import Agent, Index, QueryEngine, initialize_settings
from environ import STORAGE
from utils.chroma import chroma_client

rag_blueprint = Blueprint("rag", __name__)


@rag_blueprint.post("/<config_id>/chat")
async def rag_chat(config_id):
    query = (
        request.get_json().get("query")
        if request.is_json
        else request.form.get("query")
    )
    config: RagConfiguration = RagConfiguration.query.get_or_404(config_id)
    initialize_settings(config)
    try:
        collection = chroma_client.get_collection(config.vector_store_name)
        index = Index.from_collection(collection).index()
        query_engine = QueryEngine(index=index).query_engine()
        # vector_store_dir = os.path.join(STORAGE, f"{config.vector_store_name}")
        # index = Index.from_storage(vector_store_dir).index()
        agent = Agent(config.system_prompt, query_engine, with_context=True).agent()
        answer = await agent.run(query)
        print(f"Answer: {answer}")
    except ValueError as e:
        return jsonify(message=str(e)), 400

    return jsonify(answer=str(answer)), 200
