import os
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models import RagConfiguration
from environ import HF_HOME, EMBED_MODEL_DIR
from utils import LLM

def initialize_settings(config: RagConfiguration):

    llm = LLM(config.model_name, config.tokenizer_name)
    embed_model_name: str = config.embed_model_name
    chunk_size: int = config.chunk_size
    chunk_overlap: int = config.chunk_overlap

    temperature: float = config.temperature if config.temperature else 0.0
    top_k: int | None = config.top_k
    top_p: float | None = config.top_p

    kwargs = {"temperature": temperature} if temperature > 0 else {"do_sample": True}
    kwargs = {"top_k": top_k} if top_k else kwargs
    kwargs = {"top_p": top_p} if top_p else kwargs

    Settings.llm = HuggingFaceLLM(
        model=llm.model(),
        tokenizer=llm.tokenizer(),
        generate_kwargs=kwargs,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        cache_folder=os.path.join(HF_HOME, EMBED_MODEL_DIR) if HF_HOME else None,
    )

    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    # For testing, comment all the above and uncomment the following lines. Set up your OpenAI API key in the .env file.
    # Settings.llm = OpenAI(
    #     model="gpt-3.5-turbo",
    #     api_key=OPENAI_API_KEY,
    # )
