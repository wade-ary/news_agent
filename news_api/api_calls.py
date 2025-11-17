from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
import json

# api details https://www.newsapi.ai/intro-python
def fetch_event_registry(keyword: str, max_items: int = 10):
    """
    Run a single-query search in Event Registry for a given keyword.
    Returns a list of standardized article dictionaries.
    """

    er = EventRegistry(apiKey="pull for env")

    # Build the query: only keywords matter for now
    q = QueryArticlesIter(
        keywords = QueryItems.OR([keyword])
    )

    results = []

    # Execute query
    for art in q.execQuery(er, sortBy="rel", maxItems=max_items):
        results.append({
            "title": art.get("title"),
            "url": art.get("url"),
            "source": art.get("source", {}).get("title"),
            "published_at": art.get("dateTime"),
            "body": art.get("body")
        })

    return results
