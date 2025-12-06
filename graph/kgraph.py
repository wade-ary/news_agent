from openai import OpenAI

def compute_similarity(articles):
    """
    Build a symmetric similarity matrix combining cosine distance on embeddings
    and a lightweight topic-overlap score from the LLM.
    """
    n = len(articles)
    if n == 0:
        return []

    sim_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            art_i = articles[i]
            art_j = articles[j]

            cosine = compute_cosine(art_i.get("embedding", []), art_j.get("embedding", []))
            topic_overlap = compute_topic_overlap(
                art_i.get("full_text", "") or "",
                art_j.get("full_text", "") or "",
            )

            score = 0.5 * cosine + 0.5 * topic_overlap
            sim_matrix[i][j] = score
            sim_matrix[j][i] = score

    return sim_matrix


import math

def compute_cosine(e1, e2):
    dot = sum(a * b for a, b in zip(e1, e2))

    norm1 = math.sqrt(sum(a * a for a in e1)) if e1 else 0.0
    norm2 = math.sqrt(sum(b * b for b in e2)) if e2 else 0.0

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)

client = OpenAI()

def compute_topic_overlap(t1, t2):
    """
    Ask the LLM for a soft thematic overlap score; fall back to 0.0 on failure.
    """
    prompt = f"""
Give a score between 0 and 1 (two decimals) for how similar these articles are.

Article A:
{t1}

Article B:
{t2}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt.strip()}
            ]
        )
        raw = response.choices[0].message.content.strip()
        return float(raw)
    except Exception:
        return 0.0


def build_graph(sim_matrix, top_n=3):
    """
    Build an edge list by connecting each node to its top_n most similar peers.
    """
    n = len(sim_matrix)
    edges = []

    for i in range(n):
        sims = list(enumerate(sim_matrix[i]))
        sims = [(j, s) for j, s in sims if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = sims[:top_n]

        for j, score in top_neighbors:
            u, v = sorted((i, j))
            edges.append((u, v, score))

    return edges

from igraph import Graph

def create_graph(edges, num_articles):
    """
    Create an igraph Graph from an edge list with weights.
    """
    G = Graph()
    G.add_vertices(num_articles)
    G.add_edges([(u, v) for u, v, _ in edges])
    G.es['weight'] = [w for _, _, w in edges]
    return G
