from igraph import Graph
import leidenalg
from openai import OpenAI

def run_community(G):
    partition = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, weights='weight')

    return partition


def analyze_clusters(articles, partition):
    clusters = {}

    for cid, node_ids in enumerate(partition):
        # Gather raw text of all articles in the cluster
        cluster_texts = [articles[i]["text"] for i in node_ids]
        full_text = "\n".join(cluster_texts)

        keywords = []
        for i in node_ids:
            keywords.extend(articles[i]["keywords"])
    
        embedding = []
        for i in node_ids:
            embedding.append(articles[i]["embedding"])
        combined_summary = summarize_cluster(full_text)

        clusters[cid] = {
            "article_ids": node_ids,
            "keywords": keywords,
            "embedding": embedding,
            "combined_summary": combined_summary
        }

    return clusters

client = OpenAI()
def summarize_cluster(text):
    prompt = f"""
Give a proper summary of this combinations of articles they talk about a similar topic point out
similarities and differences.

topics:
{text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.strip()}
        ]
    )

    return response.choices[0].message.content