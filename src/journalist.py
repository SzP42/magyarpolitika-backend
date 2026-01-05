from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv
from langchain.messages import SystemMessage
import src.tools as tools
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy
import json

load_dotenv()

mistral_api = os.getenv('MISTRAL_API_KEY')


llm = ChatMistralAI(
    api_key=mistral_api,
    model="mistral-large-2512",
    temperature=0.2)

system_prompt = """
Egy újságíró vagy. Egy listát fogsz kapni JSON objektumokról, amelyek magyar hírek címeit, rövid leírásait, forrásokra és a teljes cikkre mutató linkeket tartalmaznak.
Minden cikk ugyan azt a tágabb témát fedi le, de kontextusban, narratívában, részletekben és bias-ban eltérhetnek.

A feladatod az, hogy egy részletes összefoglaló riportot írj a cikkek alapján maximum 1000 szóban, magyarul. Adj rövid, informatív, összefoglaló címet a riportnak és add vissza a megfelelő formátumban.

Hozzáférsz egy tool-hoz, amivel elő tudod hívni a teljes cikket, ha több kontextusra, véleményre van szükséged. Add meg a cikk linkjét, és visszaksz egy teljes, hosszú szöveget elemzésre.

Elemezz legalább 2 teljes cikket, de ne többet mint 5-öt. Próbálj meg különböző forrásokat választani, ha lehetséges.
"""

class Article(BaseModel):
    title: str | None = Field(description="Title of your article")
    article: str = Field(description="The full text of your article")


agent = create_agent(
    model=ChatMistralAI(model="mistral-large-2512", api_key=mistral_api, temperature=0.7, ),
    tools=[tools.read_article],
    system_prompt = system_prompt,
    response_format=ToolStrategy(Article)
)


def write_report(articles: list[dict]) -> str:
    """Writes a comprehensive report based on the provided articles."""

    articles_json = json.dumps(articles, indent=2, ensure_ascii=False)
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": articles_json}]})
    return response


async def write_reports_batch(categories_dict: dict):
    """
    Takes the whole categories dictionary and processes ALL topics in parallel.
    Input: {'Topic A': [articles...], 'Topic B': [articles...]}
    Output: {'Topic A': ArticleObject, 'Topic B': ArticleObject}
    """
    
    # Extract topics and dossiers in the same order
    topics = list(categories_dict.keys())
    topics_articles = list(categories_dict.values())
    
    # 1. Format inputs for the agent
    inputs = []
    for articles in topics_articles:
        json_str = json.dumps(articles, indent=2, ensure_ascii=False)
        inputs.append({"messages": [{"role": "user", "content": json_str}]})

    print(f"[Journalist] Async batch processing {len(inputs)} topics...")

    # 2. Execute in parallel using abatch
    # max_concurrency protects you from API rate limits
    results = await agent.abatch(inputs, config={"max_concurrency": 5})

    # 3. Map results back to topics
    final_reports = {}
    for topic, res in zip(topics, results):
        # The agent returns the structured object in this key
        final_reports[topic] = res["structured_response"]
        
    return final_reports