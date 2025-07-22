import chromadb
from llama_index.core import Document, SimpleDirectoryReader
from environ import DOCUMENTS_DIR
from utils.chroma import chroma_client


class Collection:
    def __init__(self, input_dir=DOCUMENTS_DIR, collection_name: str | None = None):
        try:
            self._collection = chroma_client.get_collection(
                collection_name or input_dir
            )

            self._documents = []

        except Exception as e:
            print("Collection not found, creating a new one.\n")
            self._collection = chroma_client.create_collection(
                collection_name or input_dir
            )

            print("Loading documents from directory...")
            self._documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
            print(f"Loaded {len(self._documents)} documents.")

    @classmethod
    def from_directory(cls, input_dir=DOCUMENTS_DIR):
        print("Loading documents from directory...")
        documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        print(f"Loaded {len(documents)} documents.")

    def documents(self) -> list[Document]:
        return self._documents

    def collection(self) -> chromadb.Collection:
        return self._collection
