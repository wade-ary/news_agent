AI news agent

1. Multi-Source, Multi-Hop Retrieval

Pulls articles from several news APIs and expands the query with LLM-generated sub-queries.

2. Cosine Similarity Filtering

Embeds all articles and keeps the top 100 most relevant to the broad topic.

3. Knowledge Graph Construction

Extracts entities and relationships from articles and connects them into a graph.

4. Topic Clustering

Groups the graph into clusters and creates short summaries and key topics for each cluster.

5. Agentic Retrieval for Questions

Finds relevant clusters and article chunks, reranks them, and answers using only retrieved evidence.

6. “Not Found” Safety Behavior

If no relevant information exists, the agent returns “not found” and offers a web search option.

7. LangGraph Orchestration

Runs the entire pipeline as a structured, inspectable workflow with modular nodes.
