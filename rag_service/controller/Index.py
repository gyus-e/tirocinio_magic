import chromadb
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from environ import VECTOR_STORE_DIR
from utils.chroma import chroma_client


class Index:
    def __init__(self, index: BaseIndex):
        self._index = index

    def index(self) -> BaseIndex:
        return self._index

    @classmethod
    def from_documents(
        cls, documents: list[Document], collection: chromadb.Collection | None = None
    ) -> "Index":
        """
        Creates an index from the provided documents.
        If a collection is provided, the index will be saved to ChromaDB.
        Otherwise, it will be created in memory.
        The user can call `persist` method to save it to storage later.
        """
        if not collection:
            print("No collection provided. Index will not be saved to ChromaDB.")
            index = VectorStoreIndex.from_documents(documents)
        else:
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=Settings.embed_model,
            )
        return cls(index)

    @classmethod
    def from_collection(cls, collection: chromadb.Collection) -> "Index":
        """
        Loads the index from a ChromaDB collection.
        """
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=Settings.embed_model,
        )
        return cls(index)

    @classmethod
    def from_storage(cls, persist_dir: str = VECTOR_STORE_DIR) -> "Index":
        """
        Loads the index from the specified storage directory.
        """
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        print("Index loaded from storage.")
        return cls(index)

    def persist(self, persist_dir: str = VECTOR_STORE_DIR):
        """
        Persists the index to the specified storage directory.
        Use if you want to retrieve it from storage later.
        """
        self._index.storage_context.persist(persist_dir=persist_dir)
        print("Index persisted to storage.")
