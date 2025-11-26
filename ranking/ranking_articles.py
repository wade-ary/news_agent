from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI


def rerank_articles(chunks, query, top_k=50):
    """
    Rerank raw text chunks using an LLM (LLMRerank).
    """

    # 1. Convert each chunk into a LlamaIndex Document
    documents = [Document(page_content=chunk) for chunk in chunks]

    # 2. Convert Documents -> Nodes
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)

    # 3. Initialize LLM-based reranker
    reranker = LLMRerank(
        top_n=min(top_k, len(nodes)),      # avoid requesting more than available
        choice_batch_size=5,
        llm=OpenAI(model="gpt-4o-mini"),   # fast + very good for reranking
    )

    # 4. Rerank nodes based on the query
    reranked_nodes = reranker.postprocess_nodes(
        nodes,
        query_str=query
    )

    # 5. Return ranked text chunks
    return [n.node.get_content() for n in reranked_nodes]