import os
from flask import Blueprint, request, jsonify
from models import RagConfiguration
from controller import Agent, Index, QueryEngine, initialize_settings
from environ import STORAGE


rag_blueprint = Blueprint("rag", __name__)


@rag_blueprint.post("/<config_id>/chat")
async def rag_chat(config_id):
    query = request.get_json().get("query") if request.is_json else request.form.get("query")
    config: RagConfiguration = RagConfiguration.query.get_or_404(config_id)
    vector_store_dir = os.path.join(STORAGE, f"vector_store_{config.config_id}")
    initialize_settings(config)
    try:
        index = Index.from_storage(vector_store_dir).index()
        query_engine = QueryEngine(index=index).query_engine()
        agent = Agent(config.system_prompt, query_engine, with_context=True).agent()
        answer = await agent.run(query)
        print(f"Answer: {answer}")
    except ValueError as e:
        return jsonify(message=str(e)), 400

    return jsonify(answer=str(answer)), 200
