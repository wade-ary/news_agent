from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding


def rerank_articles(chunks, query, top_k=50):
    """
    Retrieve the top-k relevant text chunks via vector similarity (cosine).
    """

    # 1. Convert each chunk into a LlamaIndex Document
    documents = [Document(page_content=chunk) for chunk in chunks]

    # 2. Build an in-memory vector index using OpenAI embeddings
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # 3. Retrieve top-k nodes via cosine similarity
    retriever = index.as_retriever(similarity_top_k=min(top_k, len(documents)))
    retrieved_nodes = retriever.retrieve(query)

    # 4. Return ranked text chunks
    return [n.node.get_content() for n in retrieved_nodes]