from typing import Any, Dict, List
import ast
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai_api = os.getenv("OPENAI_API_KEY")

filter_llm = ChatOpenAI(
    model="gpt-5-nano",
    api_key=openai_api,
)

tags_list = [
    "Fidesz",
    "Tisza",
    "gazdasag",
    "oktatas",
    "egeszsegugy",
    "kampany",
    "gyermekvedelem",
    "kulpolitika",
    "kozlekedes",
    "egyeb",
]

system_prompt = f"""
Egy listát fogsz kapni magyar hírek címeiről. El kell döntened, hogy a cím magyar politikával foglalkozik-e vagy sem. Csak magyar politika és gazdaság érdekel, nemzetközi, Magyarországot nem érintő hír nem.

Minden elfogadott címhez rendelj egy vagy több taget a következő listából: {tags_list}.

Válaszolj python dictionary formátumban, ahol a kulcsok a címek, az értékek pedig a tagek listája.
Csak azokat a címeket add vissza, amelyek magyar politikával foglalkoznak.
"""


def _parse_filter_response(content: Any) -> Dict[str, Any]:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Filter LLM returned non-JSON content: {content}") from None
        try:
            parsed = ast.literal_eval(text[start:end + 1])
        except (SyntaxError, ValueError):
            parsed = json.loads(text[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"Filter LLM returned JSON that is not an object: {content}")

    if "title_to_tags" in parsed:
        parsed = parsed["title_to_tags"]
        if not isinstance(parsed, dict):
            raise ValueError(f"Filter LLM title_to_tags is not an object: {content}")

    return parsed


def _parse_filter_response_by_title(content: Any, titles: List[str]) -> Dict[str, Any]:
    if not isinstance(content, str):
        return {}

    results: Dict[str, Any] = {}
    for title in titles:
        title_index = content.find(title)
        if title_index == -1:
            continue

        tags_start = content.find("[", title_index + len(title))
        if tags_start == -1:
            continue

        tags_end = content.find("]", tags_start)
        if tags_end == -1:
            continue

        try:
            results[title] = ast.literal_eval(content[tags_start:tags_end + 1])
        except (SyntaxError, ValueError):
            continue

    return results


def _normalize_filter_results(
    raw_results: Dict[str, Any],
    titles: List[str],
) -> Dict[str, List[str]]:
    allowed_titles = set(titles)
    allowed_tags = set(tags_list)
    results: Dict[str, List[str]] = {}

    for title, tags in raw_results.items():
        if title not in allowed_titles:
            continue

        if isinstance(tags, str):
            tags = [tags]

        if not isinstance(tags, list):
            continue

        valid_tags = [tag for tag in tags if tag in allowed_tags]
        if valid_tags:
            results[title] = valid_tags

    return results


def politics_filter(titles: List[str]) -> Dict[str, List[str]]:
    """Returns dict mapping titles to their tags: {title: [tag1, tag2]}"""
    print(f"[Filter] Received {titles} as titles to filter")

    if not titles:
        return {}

    print(f"[Filter] Starting politics filter on {len(titles)} titles")

    messages = [("system", system_prompt)] + [("human", title) for title in titles]
    print(f"[Filter] Messages: {messages}")

    response = filter_llm.invoke(messages)

    print(f"[Filter] LLM Response: {response.content}")

    try:
        raw_results = _parse_filter_response(response.content)
    except (json.JSONDecodeError, ValueError):
        raw_results = _parse_filter_response_by_title(response.content, titles)

    results = _normalize_filter_results(raw_results, titles)

    print(f"[Filter] Filtered {len(titles) - len(results)} out, {len(results)} passed")
    return results
