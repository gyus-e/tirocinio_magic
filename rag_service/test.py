import os
import asyncio
import torch
from controller import Agent, Index, QueryEngine, initialize_settings
from utils import LLM, Collection
from environ import DOCUMENTS_DIR, STORAGE
from test_utils import configuration, questions
from utils.chroma import chroma_client

torch.set_grad_enabled(False)

persist_dir = os.path.join(STORAGE, "test_index")


async def test():
    initialize_settings(configuration)
    collection = Collection(input_dir=DOCUMENTS_DIR, collection_name="test_collection")
    documents = collection.documents()
    chroma_collection = collection.collection()

    if documents:
        index = Index.from_documents(documents, collection=chroma_collection).index()
    else:
        index = Index.from_collection(chroma_collection).index()

    # if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    #     indexManager = Index.from_documents(documents, collection=chroma_collection)
    #     indexManager.persist()
    #     index = indexManager.index()
    # else:
    #     index = Index.from_storage(persist_dir).index()

    query_engine = QueryEngine(index).query_engine()
    agent = Agent(configuration.system_prompt, query_engine, with_context=True).agent()
    print("\n\tRAG\n")
    for i, question in enumerate(questions):
        print(f"Question {i}: {question}")
        rag_answer = await agent.run(question)
        print(f"RAG:\n{rag_answer}\n")


if __name__ == "__main__":
    asyncio.run(test())
