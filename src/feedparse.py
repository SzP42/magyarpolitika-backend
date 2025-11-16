import feedparser
import requests
from typing import List, Dict, Optional
from datetime import datetime


def parse_feed(feed_url: str, timeout: int = 10) -> Optional[Dict]:
    """
    Parse a single RSS feed from a news outlet.
    
    Args:
        feed_url: URL of the RSS feed
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing feed metadata and entries, or None if parsing failed
    """
    try:
        # Parse the feed
        feed = feedparser.parse(feed_url)
        
        # Check if parsing was successful
        if feed.bozo and feed.bozo_exception:
            print(f"Warning: Error parsing feed {feed_url}: {feed.bozo_exception}")
            return None
        
        # Extract feed information
        feed_info = {
            'feed_url': feed_url,
            'title': feed.feed.get('title', 'Unknown'),
            'link': feed.feed.get('link', ''),
            'description': feed.feed.get('description', ''),
            'entries': []
        }
        
        # Extract entries
        for entry in feed.entries:
            entry_data = {
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'description': entry.get('description', ''),
            }
            feed_info['entries'].append(entry_data)
        
        return feed_info
        
    except Exception as e:
        print(f"Error parsing feed {feed_url}: {str(e)}")
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
    """
    all_articles = []
    
    for feed_url in feed_urls:
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
    
    return all_articles
