"""
Main entry point for the application.
"""
import src.feedparse as feedparse
import json
from datetime import datetime


feed_urls = [
    "https://24.hu/feed",
    "https://telex.hu/rss",
    "https://444.hu/feed",
    "https://index.hu/24ora/rss",
    "https://www.origo.hu/publicapi/hu/rss/origo/articles",
    "https://hvg.hu/rss",
    "https://mandiner.hu/rss",
    "https://atlatszo.hu/rss"

]

def main():
    """Main function."""
    
    articles = feedparse.get_all_articles(feed_urls)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"articles_{timestamp}.txt"
    
    # Write articles to file in JSON format
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(articles)} articles to {output_file} in JSON format")


if __name__ == "__main__":
    main()


