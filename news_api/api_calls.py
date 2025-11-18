from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from newsdataapi import NewsDataAPIClient
from finlight_client import FinlightApi
from finlight_client.models import GetArticlesParams
import http.client, urllib.parse
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

# api details https://newsdata.io/blog/news-api-python-client/
def fetch_news_data(keyword: str, max_items: int = 10):
    """
    Run a single-query search in news data api for a given keyword.
    Returns a list of standardized article dictionaries.
    """
    api = NewsDataApiClient(apikey='YOUR_API_KEY')

    response = api.latest_api(q=keyword, max_results = max_items)
    results = []

    for art in response:
        results.append({
            "title": art.get("title"),
            "url": art.get("link"),
            "source": art.get("source_name"),
            "published_at": art.get("pubDate"),
            "body": art.get("body")
        })
    return results

# api details at https://docs.finlight.me/v2/rest-endpoints/
def fetch_finflight(keyword: str, max_items: int = 10):
    """
    Run a single-query search in fin flight api for a given keyword.
    Returns a list of standardized article dictionaries.
    """

    client = FinlightApi(
    config=ApiConfig(
        api_key="your_api_key"
    )
    )       

    params = GetArticlesParams(
        query=keyword,
 
    )

    response = client.articles.fetch_articles(params=params)
    results = []

    for art in response.get("articles"):
        results.append({
            "title": art.get("title"),
            "url": art.get("link"),
            "source": art.get("source_name"),
            "published_at": art.get("publishDate"),
            "body": art.get("summary")
        })
    
    return results

def fetch_the_news_api(keyword : str, max_items: int = 10):
    """
    Run a single-query search in fin flight api for a given keyword.
    Returns a list of standardized article dictionaries.
    """
    conn = http.client.HTTPSConnection('api.thenewsapi.com')

    params = urllib.parse.urlencode({
        'api_token': 'YOUR_API_TOKEN',
        'categories': keyword,
        'limit': max_items,
        })

    conn.request('GET', '/v1/news/all?{}'.format(params))

    res = conn.getresponse()
    data = res.read()

    print(data.decode('utf-8'))

    results = []
    
    for art in data:
        results.append({
            "title": art.get("title"),
            "url": art.get("url"),
            "source": art.get("source"),
            "published_at": art.get("published_at"),
            "body": art.get("description")
        })
    
    return results
