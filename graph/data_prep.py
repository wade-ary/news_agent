from newspaper import Article
import json



def get_full_texts(articles):
    # 1. Deduplicate by URL
    unique = set()
    unique_articles = []
    for article in articles:
        url = article.get("url")
        if url not in unique:
            unique.add(url)
            unique_articles.append(article)

    # 2. Scrape & add full_text field
    for article in unique_articles:
        url = article["url"]
        try:
            article_text = scrape_url(url)
        except:
            article_text = None

        # Add the new field to the article dict
        article["full_text"] = article_text

    return unique_articles

def scrape_url(url):
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = art.text
    except:
        return None

    if len(text) < 300:
        return None

    return text

from openai import OpenAI

client = OpenAI()

def get_embeddings(articles):
    for article in articles:
        text = article.get("full_text") or ""
        
        # Call OpenAI embeddings endpoint
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        
        # Save embedding vector to article dict
        article["embedding"] = response.data[0].embedding

    return articles

def get_topics(articles):

    for article in articles:
        topics_json = extract_topics(article["full_text"])
        article["topics"] = json.loads(topics_json)

    return articles


