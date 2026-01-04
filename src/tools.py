from langchain.tools import tool
from firecrawl import Firecrawl
import os
from dotenv import load_dotenv

load_dotenv()
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

firecrawl = Firecrawl(api_key=firecrawl_api_key)

@tool
def read_article(link: str):
    """
    Scrapes the full text content of a news article given its link.
    Use this to get detailed context before writing the report.
    """
    
    # Define the schema for the data you want to extract
    # This matches the "schema" part of your payload
    extraction_schema = {
        "type": "object",
        "required": ['article_text'],
        "properties": {
            "article_text": {
                "type": "string"
            }
        }
    }

    try:
        result = firecrawl.scrape(
            url=link,
            formats=[
                {
                    "type": "json", 
                    "schema": extraction_schema,
                    "prompt": "You will receive an article. Extract the core useful part of the page"
                }
            ],
        )
        return result.json['article_text']
    except Exception as e:
        return f"Error scraping article: {str(e)}"

