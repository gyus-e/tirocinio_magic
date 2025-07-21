import os
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models import RagConfiguration
from utils import LLM
from environ import HF_HOME, EMBED_MODEL_DIR


def initialize_settings(config: RagConfiguration):

    model_name = config.model_name
    llm = LLM(model_name=model_name)
    embed_model_name: str = config.embed_model_name
    chunk_size: int = config.chunk_size
    chunk_overlap: int = config.chunk_overlap

    temperature: float = config.temperature
    top_k: int | None = config.top_k
    top_p: float | None = config.top_p

    kwargs = {"temperature": temperature} if temperature > 0 else {"do_sample": True}
    kwargs = {"top_k": top_k} if top_k else kwargs
    kwargs = {"top_p": top_p} if top_p else kwargs

    Settings.llm = HuggingFaceLLM(
        model=llm.model(),
        tokenizer=llm.tokenizer(),
        # context_window=CONTEXT_WINDOW if CONTEXT_WINDOW else DEFAULT_CONTEXT_WINDOW,
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
