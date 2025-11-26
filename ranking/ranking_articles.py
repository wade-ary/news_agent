import os
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization= os.getenv("OPENAI_ORG_ID"))

embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key= os.getenv("OPENAI_API_KEY"),
    project= os.getenv("OPENAI_PROJECT_ID"),
    organization= os.getenv("OPENAI_ORG_ID"),
    embed_batch_size=1
)

def rerank_chunks(
    chunks, 
    query, 
    top_k=50, ):
    """
    Rerank chunks using a basic HuggingFace embedding model.
    Default: BAAI/bge-small-en-v1.5 (lightweight, reliable, and fully supported).
    """

    # 1. Convert raw text → Documents
    documents = [Document(text=chunk) for chunk in chunks]

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)



    # 4. Build vector index
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    # 5. Retrieve top_k similar chunks
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    results = retriever.retrieve(query)

    # 6. Extract ranked content
    return [res.node.get_content() for res in results]