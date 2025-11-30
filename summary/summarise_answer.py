import math
from openai import OpenAI
import json
client = OpenAI()
import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_k_clusters(query_embedding, clusters, k=8):
    scored = []

    for c in clusters:
        score = cosine_sim(query_embedding, c["summary_embedding"])
        scored.append((score, c))

    # sort by descending similarity
    scored.sort(key=lambda x: x[0], reverse=True)

    # return only cluster dicts
    return [c for score, c in scored[:k]]


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
