import feedparser
import re
from typing import List, Dict, Optional
from datetime import datetime
from bs4 import BeautifulSoup 


def clean_text(text: str) -> str:
    """
    Universal cleaner for RSS feeds.
    Handles: HTML tags, CDATA artifacts, specific footers, and whitespace.
    """
    if not text:
        return ""

    # 1. Remove HTML tags (p, a, img, etc.)
    # BeautifulSoup handles HTML entities (&amp;) and tags automatically.
    soup = BeautifulSoup(text, "html.parser")
    clean = soup.get_text(separator=" ")

    # 2. Remove "The post ... first appeared on ..." (24.hu specific artifact)
    clean = re.sub(r"The post.*?first appeared on.*?24\.hu\.?", "", clean, flags=re.IGNORECASE)

    # 3. Remove "Source" links at the end (Common in Atlatszo/others)
    clean = re.sub(r"\s*Source\s*$", "", clean, flags=re.IGNORECASE)

    # 4. Collapse multiple spaces/newlines into single space
    clean = " ".join(clean.split())

    return clean


def parse_feed(feed_url: str, timeout: int = 10) -> Optional[Dict]:
    """
    Parse a single RSS feed from a news outlet.
    
    Args:
        feed_url: URL of the RSS feed
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing feed metadata and entries, or None if parsing failed
    """
    print(f"[FeedParse] Parsing feed: {feed_url}")
    try:
        # Parse the feed
        feed = feedparser.parse(feed_url)
        
        # Check if parsing was successful
        if feed.bozo and feed.bozo_exception:
            print(f"[FeedParse] Warning: Error parsing feed {feed_url}: {feed.bozo_exception}")
            return None
        
        # Extract feed information
        feed_info = {
            'feed_url': feed_url,
            'title': feed.feed.get('title', 'Unknown'),
            'link': feed.feed.get('link', ''),
            'description': feed.feed.get('description', ''),
            'entries': []
        }

        for entry in feed.entries:
            # Get raw text
            raw_title = entry.get('title', '')
            raw_desc = entry.get('description', '')

            # Clean_text
            clean_title = clean_text(raw_title)
            clean_desc = clean_text(raw_desc)

            if len(clean_desc) > 1000:
                clean_desc = clean_desc[:1000].rsplit(' ', 1)[0] + "..."

            entry_data = {
                'title': clean_title,
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'description': clean_desc
            }

            feed_info['entries'].append(entry_data)

        print(f"[FeedParse] Successfully parsed '{feed_info['title']}': {len(feed_info['entries'])} articles found")
        return feed_info
        
    except Exception as e:
        print(f"[FeedParse] Error parsing feed {feed_url}: {str(e)}")
        return None
           

def get_all_articles(feed_urls: List[str], timeout: int = 10) -> List[Dict]:
    """
    Parse all RSS feeds in a list and return a flattened list of all articles.
    
    Args:
        feed_urls: List of RSS feed URLs
        timeout: Request timeout in seconds for each feed
        
    Returns:
        List of article dictionaries, each with source information
        [{
        'source': 'source_name',
        'source_url': 'source_url',
        'title': 'article_title',
        'link': 'article_link',
        'published': 'article_published',
        'description': 'article_description'
        }, ... 
        ]

        in clustering, after embedding a vector will be added as well ('vector': [float, float])
    """
    print(f"[FeedParse] Starting to parse {len(feed_urls)} feeds...")
    all_articles = []
    
    for i, feed_url in enumerate(feed_urls, 1):
        print(f"[FeedParse] Processing feed {i}/{len(feed_urls)}...")
        feed_data = parse_feed(feed_url, timeout)
        if feed_data:
            source = feed_data['title']
            link = feed_data['link']
            for entry in feed_data['entries']:
                article = {
                    'source': source,
                    'source_url': link,
                    **entry
                }
                all_articles.append(article)
            print(f"[FeedParse] Added {len(feed_data['entries'])} articles from '{source}'")
        else:
            print(f"[FeedParse] Failed to parse feed {i}/{len(feed_urls)}")
    
    print(f"[FeedParse] Feed parsing complete: {len(all_articles)} total articles")
    return all_articles
