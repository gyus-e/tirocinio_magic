import chromadb
from llama_index.core import Document, SimpleDirectoryReader
from environ import DOCUMENTS_DIR
from utils.chroma import chroma_client


class Collection:
    def __init__(self, input_dir=DOCUMENTS_DIR, collection_name: str | None = None):
        print("Loading documents from directory...")
        self._documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        print(f"Loaded {len(self._documents)} documents.")
        try:
            self._collection = chroma_client.get_collection(
                collection_name or input_dir
            )
            self._is_new = False
        except Exception as e:
            self._collection = chroma_client.create_collection(
                collection_name or input_dir
            )
            self._is_new = True

    def documents(self) -> list[Document]:
        return self._documents

    def collection(self) -> chromadb.Collection:
        return self._collection

    def is_new(self) -> bool:
        return self._is_new
