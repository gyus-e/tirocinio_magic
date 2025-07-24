import chromadb

chroma_client = chromadb.HttpClient()

try:
    chroma_client.delete_collection("test_collection")
    print("Test collection deleted successfully. Remaining collections:")
except Exception as e:
    print(f"Error deleting test collection: {e}")

print(chroma_client.list_collections())
