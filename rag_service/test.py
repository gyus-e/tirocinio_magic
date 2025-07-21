import os
import asyncio
from controller import Agent, Index, QueryEngine, initialize_settings
from utils import LLM, Collection
from environ import DOCUMENTS_DIR, STORAGE
from test.configuration_mock import configuration
from test.questions_mock import questions

persist_dir = os.path.join(STORAGE, "test_index")

async def test():
    initialize_settings(configuration)
    documents = Collection(input_dir=DOCUMENTS_DIR).documents()
    
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        index = Index.from_documents(documents).persist(persist_dir).index()
    else:
        index = Index.from_storage(persist_dir).index()

    query_engine = QueryEngine(index).query_engine()
    agent = Agent(configuration.system_prompt, query_engine, with_context=True).agent()
    print("\n\tRAG\n")
    for i, question in enumerate(questions):
        print(f"Question {i}: {question}")
        rag_answer = await agent.run(question)
        print(f"Answer: {rag_answer}\n")


if __name__ == "__main__":
    asyncio.run(test())
