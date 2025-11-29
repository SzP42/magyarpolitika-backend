"""
Main entry point for the application.
"""
import src.feedparse as feedparse
import json
from datetime import datetime
import time
import src.first_filter as filter


feed_urls = [
    # "https://444.hu/feed",
    # "https://24.hu/feed",
    "https://telex.hu/rss",
    # "https://index.hu/24ora/rss",
    # "https://www.origo.hu/publicapi/hu/rss/origo/articles",
    # # "https://hvg.hu/rss",
    # # "https://mandiner.hu/rss",
    # # "https://atlatszo.hu/rss"

]

def main():
    """Main function."""
    
    articles = feedparse.get_all_articles(feed_urls)
    print(len(articles))
    
    filter_results = filter.politics_filter(articles)
    for k, i in enumerate(filter_results[:20]):
        print("---------------")
        for j in filter_results[0].keys():
            print(f"{j}: {filter_results[k][j]}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"{'='*50}")


