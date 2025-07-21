from llama_index.core import Document, SimpleDirectoryReader
from environ import DOCUMENTS_DIR


class Collection:
    def __init__(self, input_dir=DOCUMENTS_DIR):
        print("Loading documents from directory...")
        self._documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        print(f"Loaded {len(self._documents)} documents.")

    def documents(self) -> list[Document]:
        return self._documents
