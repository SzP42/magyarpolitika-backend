from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai_api = os.getenv('OPENAI_API_KEY')

filter_llm = ChatOpenAI(
    model="gpt-5-nano",
    api_key=openai_api,
)

tags_list = ["Fidesz", "Tisza", "gazdasag", "oktatas", "egeszsegugy", "kampany", "gyermekvedelem", "kulpolitika", "kozlekedes", "egyeb"]

system_prompt = f"""
Egy listát fogsz kapni magyar hírek címeiről. El kell döntened, hogy a cím magyar politikával foglalkozik-e vagy sem. Csak magyar politika és gazdaság érdekel, nemzetközi, Magyarországot nem érintő hír nem.

Minden elfogadott címhez rendelj egy vagy több taget a következő listából: {tags_list}.

Válaszolj python dictionary formátumban, ahol a kulcsok a címek, az értékek pedig a tagek listája.
Csak azokat a címeket add vissza, amelyek magyar politikával foglalkoznak.
"""

def politics_filter(titles: List[str]) -> Dict[str, List[str]]:
    print(f"[Filter] Received {titles} as titles to filter")

    """Returns dict mapping titles to their tags: {title: [tag1, tag2]}"""
    print(f"[Filter] Starting politics filter on {len(titles)} titles")

    messages = [("system", system_prompt)] + [("human", title) for title in titles]
    print(f"[Filter] Messages: {messages}")

    response = filter_llm.invoke(messages)

    print(f"[Filter] LLM Response: {response.content}")

    results = eval(response.content)

    print(f"[Filter] Filtered {len(titles) - len(results)} out, {len(results)} passed")
    return results
