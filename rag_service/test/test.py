import asyncio
from controller import Agent, Index, QueryEngine, initialize_settings
from utils import LLM, Collection
from environ import DOCUMENTS_DIR
from .configuration_mock import configuration
from .questions_mock import questions

documents = None
llm = LLM(configuration.model_name)
initialize_settings(configuration)

async def test():
    documents = Collection(input_dir=DOCUMENTS_DIR).documents()
    index = Index.from_documents(documents).index()
    query_engine = QueryEngine(index).query_engine()
    agent = Agent(configuration.system_prompt, query_engine, with_context=True).agent()
    print("\n\tRAG\n")
    for i, question in enumerate(questions):
        print(f"Question {i}: {question}")
        rag_answer = await agent.run(question)
        print(f"Answer: {rag_answer}\n")


if __name__ == "__main__":
    asyncio.run(test())
