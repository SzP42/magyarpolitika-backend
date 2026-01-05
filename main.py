"""
Main entry point for the application.
"""
from sys import exception
from typing import Any
import re
import unicodedata
import src.feedparse as feedparse
import time
import src.filter as filter
import src.clustering as clustering 
import src.vstorage as vstorage
import src.journalist as journalist
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import asyncio
import json
import pprint

load_dotenv()
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY: str = os.getenv("SUPABASE_SECRET_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

feed_urls = [
    "https://444.hu/feed",
    "https://24.hu/feed",
    "https://telex.hu/rss",
    "https://index.hu/24ora/rss",
    "https://www.origo.hu/publicapi/hu/rss/origo/articles",
    "https://hvg.hu/rss",
    "https://mandiner.hu/rss", 
    "https://atlatszo.hu/rss",
    "https://www.portfolio.hu/rss/unios-forrasok.xml",
    "https://www.portfolio.hu/rss/ingatlan.xml",
    "https://www.portfolio.hu/rss/gazdasag.xml",


]

def slugify(text: str) -> str:
    """
    Converts a news title into a Pinecone-safe namespace string.
    """
    # Normalize unicode characters to decompose combined characters (like 'ö' to 'o' + '¨')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remove everything that isn't a word character, space, or hyphen
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    
    # Replace spaces and multiple hyphens with a single hyphen
    text = re.sub(r'[-\s]+', '-', text)
    
    # Limit to 60 characters (Pinecone namespaces have length limits)
    return text[:60]


async def main():
    """Main function."""
    
    raw_articles = feedparse.get_all_articles(feed_urls)
    
    # 1. Fetch topics from Pinecone
    topic_memory_dict = vstorage.query_topic_memory()

    topics_before_clustering = set(topic_memory_dict.keys())

    raw_categories_dict = clustering.assign_topics(articles=raw_articles, known_topics=topic_memory_dict)
    
    filter_results = filter.politics_filter(list(raw_categories_dict.keys()))

    categories_dict = {k: v for k, v in raw_categories_dict.items() if k in filter_results}
    
    # upload topics to Pinecone, get historical data for journalist agent, 
    for topic_name, articles in categories_dict.items():
        slug_id = slugify(topic_name)
        is_known = topic_name in topics_before_clustering

        print(f"[Main] topic name {topic_name}")
        
        if not is_known:
            # Brand new topic, wasn't in the DB before clustering 
            vstorage.upsert_to_namespace(topic_name, slug_id, articles, save_to_memory=True)
        else: 
            # It was already in Pinecone at the start of the script.

            hist_articles = vstorage.get_data_for_namespace(slug_id)

            vstorage.upsert_to_namespace(topic_name, slug_id, articles, save_to_memory=False)

            unique_history = [h for h in hist_articles if not any(h['link'] == a['link'] for a in categories_dict[topic_name])]

            categories_dict[topic_name].extend(unique_history)

            print(f"[Main] Merged {len(unique_history)} historical articles for context.")

    
        clean_articles = [{k: v for k, v in article.items() if k != 'vector'} for article in categories_dict[topic_name]]
        categories_dict[topic_name] = clean_articles

    # Dump the dict in as is
    reports_map = await journalist.write_reports_batch(categories_dict)

    final_values = [{'title': report.title, 'article': report.article, 'sources': [{'title': a['title'], 'link': a['link']} for a in categories_dict[topic]]} for topic, report in reports_map.items()]

    try:
        response = (
        supabase.table("articles")
        .insert(final_values)
        .execute()
    )
        print(f"[Main] Supabase insert response: {response}")
    except Exception as exception:
        print(f"[Main] Supabase insert exception: {exception}")

    # Handle results
    for topic, report in reports_map.items():
        print(f"\n{'='*30}")
        print(f"TOPIC: {topic}")
        print(f"TITLE: {report.title}")
        print(f"REPORT:\n{report.article[:500]}")

    

    
def test():
    pass


if __name__ == "__main__":
    start_time = time.perf_counter()    
    asyncio.run(main())
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"{'='*50}")
