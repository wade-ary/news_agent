"""
LangGraph-powered agentic workflow for the news assistant.

Pipeline:
1) Fetch articles from multiple providers (news_api/api_calls).
2) Enrich articles with full text, embeddings, and topics.
3) Build similarity graph and cluster articles.
4) Rank clusters and draft an answer (summary/summarise_answer.py).
5) Optional refinement loop using a follow-up query.

Note: This module relies on API keys for the underlying providers and OpenAI.
Ensure the relevant environment variables/config are set before execution.
"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph import data_prep, graph_analysis, kgraph
from news_api import api_calls
from ranking.ranking_articles import rerank_articles
from summary import summarise_answer


class AgentState(TypedDict, total=False):
    topic: str
    refine_query: Optional[str]
    raw_articles: List[Dict[str, Any]]
    enriched_articles: List[Dict[str, Any]]
    sim_matrix: List[List[float]]
    edges: List[Any]
    igraph: Any
    clusters: List[Dict[str, Any]]
    ranked_clusters: List[Dict[str, Any]]
    answer: str


# --- LangGraph nodes ----------------------------------------------------- #


def fetch_articles(state: AgentState) -> AgentState:
    topic = state.get("topic") or ""
    fetchers = [
        ("event_registry", api_calls.fetch_event_registry),
        ("news_data", api_calls.fetch_news_data),
        ("finflight", api_calls.fetch_finflight),
        ("the_news_api", api_calls.fetch_the_news_api),
    ]

    collected: List[Dict[str, Any]] = []
    for name, fn in fetchers:
        try:
            collected.extend(fn(topic, max_items=10))
        except Exception as exc:  # noqa: BLE001 - surface provider issues but keep going
            print(f"[fetch:{name}] skipped due to error: {exc}")

    state["raw_articles"] = collected
    return state


def enrich_articles(state: AgentState) -> AgentState:
    articles = state.get("raw_articles", [])
    if not articles:
        state["enriched_articles"] = []
        return state

    enriched = data_prep.get_full_texts(articles)
    enriched = data_prep.get_embeddings(enriched)
    enriched = data_prep.get_topics(enriched)

    # Align with graph_analysis expectations.
    for art in enriched:
        art["text"] = art.get("full_text") or art.get("body") or ""
        topics = art.get("topics") or []
        art["keywords"] = topics if isinstance(topics, list) else topics

    state["enriched_articles"] = enriched
    return state


def build_similarity_graph(state: AgentState) -> AgentState:
    arts = state.get("enriched_articles", [])
    sim_matrix = kgraph.compute_similarity(arts)
    edges = kgraph.build_graph(sim_matrix, top_n=3)
    igraph_obj = kgraph.create_graph(edges, num_articles=len(arts)) if arts else None

    state["sim_matrix"] = sim_matrix
    state["edges"] = edges
    state["igraph"] = igraph_obj
    return state


def cluster_and_summarize(state: AgentState) -> AgentState:
    G = state.get("igraph")
    arts = state.get("enriched_articles", [])
    if not G or not arts:
        state["clusters"] = []
        return state

    partition = graph_analysis.run_community(G)
    clusters_dict = graph_analysis.analyze_clusters(arts, partition)

    clusters: List[Dict[str, Any]] = []
    for cid, data in clusters_dict.items():
        clusters.append(
            {
                "cid": cid,
                "summary": data.get("combined_summary", ""),
                "articles": [arts[i] for i in data.get("article_ids", [])],
                "keywords": data.get("keywords", []),
                "embedding": data.get("summary_embedding"),
            }
        )

    state["clusters"] = clusters
    return state


def rank_clusters(state: AgentState) -> AgentState:
    clusters = state.get("clusters", [])
    topic = state.get("topic") or ""

    if not clusters:
        state["ranked_clusters"] = []
        return state

    chunks = [c.get("summary", "") for c in clusters]
    try:
        ordered = rerank_articles(chunks, topic, top_k=len(chunks))
        summary_to_cluster = {c.get("summary", ""): c for c in clusters}

        ranked: List[Dict[str, Any]] = []
        for summary in ordered:
            cluster = summary_to_cluster.get(summary)
            if cluster and cluster not in ranked:
                ranked.append(cluster)

        # Append any missing clusters to preserve completeness.
        for c in clusters:
            if c not in ranked:
                ranked.append(c)

        state["ranked_clusters"] = ranked
    except Exception as exc:  # noqa: BLE001 - fallback to original order
        print(f"[rank_clusters] fallback due to error: {exc}")
        state["ranked_clusters"] = clusters

    return state


def draft_response(state: AgentState) -> AgentState:
    clusters = state.get("ranked_clusters") or state.get("clusters") or []
    topic = state.get("topic") or ""

    if not clusters:
        state["answer"] = "No relevant articles found."
        return state

    top_clusters = summarise_answer.retrieve_top_k_clusters(
        topic, clusters, k=min(8, len(clusters))
    )
    filtered = summarise_answer.llm_filter_clusters(topic, top_clusters)
    answer = summarise_answer.draft_answer(topic, filtered)
    state["answer"] = answer
    return state


def refine_response(state: AgentState) -> AgentState:
    refine_query = state.get("refine_query")
    if not refine_query:
        return state

    clusters = state.get("ranked_clusters") or state.get("clusters") or []
    filtered = summarise_answer.llm_filter_clusters(refine_query, clusters)
    answer = summarise_answer.draft_answer(refine_query, filtered)
    state["answer"] = answer
    state["refine_query"] = None  # prevent loops
    return state


# --- LangGraph wiring ---------------------------------------------------- #


def _should_refine(state: AgentState):
    return "refine_answer" if state.get("refine_query") else END


def build_graph() -> Any:
    """
    Build and compile the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_articles", fetch_articles)
    workflow.add_node("enrich_articles", enrich_articles)
    workflow.add_node("build_graph", build_similarity_graph)
    workflow.add_node("cluster_and_summarize", cluster_and_summarize)
    workflow.add_node("rank_clusters", rank_clusters)
    workflow.add_node("draft_response", draft_response)
    workflow.add_node("refine_answer", refine_response)

    workflow.set_entry_point("fetch_articles")
    workflow.add_edge("fetch_articles", "enrich_articles")
    workflow.add_edge("enrich_articles", "build_graph")
    workflow.add_edge("build_graph", "cluster_and_summarize")
    workflow.add_edge("cluster_and_summarize", "rank_clusters")
    workflow.add_edge("rank_clusters", "draft_response")
    workflow.add_conditional_edges("draft_response", _should_refine, ["refine_answer", END])
    workflow.add_edge("refine_answer", END)

    return workflow.compile(checkpointer=MemorySaver())


def run_once(topic: str, refine_query: Optional[str] = None) -> AgentState:
    """
    Convenience helper to run the full workflow once.
    """
    graph = build_graph()
    result = graph.invoke({"topic": topic, "refine_query": refine_query})
    return result  # type: ignore[return-value]


__all__ = [
    "AgentState",
    "build_graph",
    "run_once",
]

