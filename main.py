"""
Main entry point for the application.
"""
from typing import Any
import re
import unicodedata
import src.feedparse as feedparse
import time
import src.filter as filter
import src.clustering as clustering 
import src.vstorage as vstorage

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index = pc.Index(host="https://politics-app-551uxi4.svc.aped-4627-b74a.pinecone.io")

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


def main():
    """Main function."""

    # Optional: Debug list
    # for ids in index.list(namespace='topic_memory'):
    #     print(ids)
    
    raw_articles = feedparse.get_all_articles(feed_urls)
    
    # 1. Fetch topics from Pinecone
    topic_memory_dict = vstorage.query_topic_memory()

    # 2. FIX: Create a set of keys strictly for what is ALREADY in the DB
    # We use a set for faster lookups and to detach from the dictionary object
    known_topics_snapshot = set(topic_memory_dict.keys())

    # 3. Clustering (This function appears to update topic_memory_dict in place)
    raw_categories_dict = clustering.assign_topics(articles=raw_articles, known_topics=topic_memory_dict)
    
    filter_results = filter.politics_filter(list(raw_categories_dict.keys()))

    categories_dict = {k: v for k, v in raw_categories_dict.items() if k in filter_results}

    print(f"[Main] categories_dict keys: {categories_dict.keys()}")
    
    # This print proved the dict was modified:
    # print(f"[Main] topic memory keys {topic_memory_dict.keys()}") 

    for topic_name, articles in categories_dict.items():
        slug_id = slugify(topic_name)

        print(f"[Main] topic name {topic_name}")
        
        # 4. FIX: Check against the SNAPSHOT, not the modified dictionary
        is_known = topic_name in known_topics_snapshot
        print(f"[Main] topic_name in KNOWN_SNAPSHOT: {is_known}")

        if not is_known:
            # It's not in the snapshot, so it's new. Save to Pinecone 'topic_memory'.
            vstorage.upsert_to_namespace(topic_name, slug_id, articles, save_to_memory=True)
        else: 
            # It was already in Pinecone at the start of the script.
            vstorage.upsert_to_namespace(topic_name, slug_id, articles, save_to_memory=False)


def test():
    topic_memory_dict = vstorage.query_topic_memory()
    print(topic_memory_dict.keys())
        
    
if __name__ == "__main__":
    start_time = time.perf_counter()    
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"{'='*50}")
