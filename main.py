"""
Main entry point for the application.
"""
from typing import Any
import src.feedparse as feedparse
import time
import src.first_filter as filter
import src.clustering as clustering 


feed_urls = [
    "https://444.hu/feed",
    "https://24.hu/feed",
    "https://telex.hu/rss",
    "https://index.hu/24ora/rss",
    "https://www.origo.hu/publicapi/hu/rss/origo/articles",
    "https://hvg.hu/rss",
    "https://mandiner.hu/rss",
    "https://atlatszo.hu/rss",

]

def main():
    """Main function."""
    
    raw_articles = feedparse.get_all_articles(feed_urls)
    
    filter_results = filter.politics_filter(raw_articles, use_smart_model=False)
    
    clustered_articles, known_topics = clustering.assign_topics(articles=filter_results)
    
    # Print each article
    for article in clustered_articles:
        print(f"Title: {article.get('title', 'No title')}")
        print(f"Source: {article.get('source', 'Unknown')}")
        if article.get('published'):
            print(f"Published: {article.get('published')}")
        if article.get('link'):
            print(f"Link: {article.get('link')}")
        if article.get('description'):
            print(f"Description: {article.get('description')}")
        if 'Category' in article:
            print(f"Category: {article.get('Category')}")
        print()

    

if __name__ == "__main__":
    start_time = time.perf_counter()    
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"{'='*50}")


