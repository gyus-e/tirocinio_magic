import os
from environ import STORAGE
from models import CagConfiguration
from utils import Collection, LLM
from .cag_context import build_cag_context
from . import create_kv_cache, save_cache

def initialize_cache(configuration: CagConfiguration, llm: LLM):
    cache_name = f"{configuration.cache_name}.cache"
    cache_path = os.path.join(STORAGE, cache_name)
    
    if not os.path.exists(cache_path):
        documents = Collection().documents()

        document_texts = [doc.text for doc in documents]
        cag_prompt = build_cag_context(
            system_prompt=configuration.system_prompt,
            document_texts=document_texts,
        )

        cache = create_kv_cache(
            model=llm.model(),
            tokenizer=llm.tokenizer(),
            prompt=cag_prompt,
        )

        save_cache(cache, cache_name=cache_name, storage=STORAGE)

    else:
        print(f"Cache already exists at {cache_path}, skipping initialization.")

    return cache_path
