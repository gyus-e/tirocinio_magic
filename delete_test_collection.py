import chromadb

chroma_client = chromadb.HttpClient()

try:
    chroma_client.delete_collection("test_collection")
    print("Test collection deleted successfully. Remaining collections:")
except Exception as e:
    print(f"Warning: {e}")

print(chroma_client.list_collections())
