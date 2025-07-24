import chromadb

# chroma_client = chromadb.HttpClient() # Use with ollama server running on docker
chroma_client = chromadb.EphemeralClient()  # Use when running on colab
