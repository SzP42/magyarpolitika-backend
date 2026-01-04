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

system_prompt = """You are a journalist. You will get a list of JSON objects about articles containing Hungarian news headlines, brief descriptions, links to sources, and to the full article itself. 
All articles cover the same broader topic but may differ in nuance, context details, and bias
Your job is to write a comprehensive report based on the articles. Give a short summarizing title to your piece and return it in the appropriate format. 
You have access to a tool that will get the full article for you if you decide you want more context from one. Provide the link, and you'll get back a long text for you to analyse. 
Analyse at least 2 full articles, but not more than 5. Pick different sources when possible.
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
