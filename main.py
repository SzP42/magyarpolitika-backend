"""
Main entry point for the application.
"""
from typing import Any
import src.feedparse as feedparse
import time
import src.filter as filter
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
    
    clustered_articles, known_topics, categories_dict = clustering.assign_topics(articles=raw_articles)
    
    filter_results = filter.politics_filter(list(categories_dict.keys()))

    print(filter_results)

#     # Print categories_dict in a readable format
#     print("\n" + "="*80)
#     print("CATEGORIES SUMMARY")
#     print("="*80)
    
#     for topic_name, articles in categories_dict.items():
#         print(f"\n{'─'*80}")
#         print(f"📰 TOPIC: {topic_name}")
#         print(f"   Articles: {len(articles)}")
#         print(f"{'─'*80}")
        
#         for idx, article in enumerate(articles, 1):
#             print(f"\n  [{idx}] {article.get('title', 'N/A')}")
#             print(f"      Source: {article.get('source', 'N/A')}")
#             print(f"      Published: {article.get('published', 'N/A')}")
#             if article.get('description'):
#                 desc = article['description'][:100] + "..." if len(article.get('description', '')) > 100 else article.get('description', '')
#                 print(f"      Description: {desc}")
#             print(f"      Link: {article.get('link', 'N/A')}")
    
#     print(f"\n{'='*80}")
#     print(f"Total Categories: {len(categories_dict)}")
#     print(f"Total Articles: {sum(len(articles) for articles in categories_dict.values())}")
#     print(f"{'='*80}\n")
    
if __name__ == "__main__":
    start_time = time.perf_counter()    
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"{'='*50}")
