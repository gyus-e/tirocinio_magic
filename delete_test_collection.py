import chromadb

chroma_client = chromadb.HttpClient()
chroma_client.delete_collection("test_collection")

print("Test collection deleted successfully. Remaining collections:")
print(chroma_client.list_collections())
