from openai import OpenAI

def compute_similarity(articles):
    n = len(articles)

    
    sim_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    # loop over all pairs
    for i in range(n):
        for j in range(i + 1, n):   # upper triangle only
            art_i = articles[i]
            art_j = articles[j]

      
            cosine = compute_cosine(art_i["embedding"], art_j["embedding"])
            topic_overlap = compute_topic_overlap(art_i["topics"], art_j["topics"])
      

       
            score = 0.5 * cosine + 0.5 * topic_overlap

            sim_matrix[i][j] = score
            sim_matrix[j][i] = score

    return sim_matrix


import math

def compute_cosine(e1, e2):
   
    dot = sum(a * b for a, b in zip(e1, e2))

    
    norm1 = math.sqrt(sum(a * a for a in e1))
    norm2 = math.sqrt(sum(b * b for b in e2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)

client = OpenAI()

def compute_topic_overlap(t1, t2):
    prompt = f"""
Give a score between 0 and 1 2 decimal places for the overlap in topics using semantics

topics:
{t1, t2}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.strip()}
        ]
    )

    return response.choices[0].message.content
