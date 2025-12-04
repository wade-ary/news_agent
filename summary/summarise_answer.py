import json
from openai import OpenAI
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI as LlamaOpenAI

client = OpenAI()


def retrieve_top_k_clusters(query, clusters, k=8):
    """
    Use LlamaIndex's LLMRerank to order clusters by relevance to the query.
    """
    # Build Documents from cluster summaries so metadata travels through rerank
    documents = [
        Document(text=c["summary"], metadata={"cid": c["cid"]})
        for c in clusters
    ]

    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)

    reranker = LLMRerank(
        top_n=min(k, len(nodes)),
        choice_batch_size=5,
        llm=LlamaOpenAI(model="gpt-4o-mini"),
    )
    reranked = reranker.postprocess_nodes(nodes, query_str=query)

    cid_to_cluster = {c["cid"]: c for c in clusters}
    ordered = []
    for n in reranked:
        cid = n.node.metadata.get("cid")
        if cid in cid_to_cluster:
            ordered.append(cid_to_cluster[cid])

    # If reranker omitted some clusters, append remaining to preserve coverage
    if len(ordered) < len(clusters):
        remaining = [c for c in clusters if c["cid"] not in {cl["cid"] for cl in ordered}]
        ordered.extend(remaining)

    return ordered[:k]


def llm_filter_clusters(query, candidate_clusters):
    cluster_blocks = []
    for c in candidate_clusters:
        cluster_blocks.append(
            f"[CID {c['cid']}]\nSummary: {c['summary']}\n"
        )

    prompt = f"""
User query: "{query}"

Below are the candidate story clusters. 
Each has a summary. Rank them by relevance to the query and return only the 
relevant cluster IDs as a JSON list.

Clusters:
{''.join(cluster_blocks)}

Return ONLY JSON list of CIDs.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.strip()}
        ]
    )
    selected_cids = json.loads(response)

    # filter the cluster dicts
    return [c for c in candidate_clusters if c["cid"] in selected_cids]
