from chromadb import PersistentClient
from langchain_chroma import Chroma

def get_vectorstore(embeddings):
    # Client baru (Chroma v0.5+)
    client = PersistentClient(path="./vectorstore")

    # Create or load collection
    vectorstore = Chroma(
        client=client,
        collection_name="docs",
        embedding_function=embeddings,
    )

    return vectorstore
