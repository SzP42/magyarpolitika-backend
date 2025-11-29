import json
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from typing import List, Dict

smart_model = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
dumb_model = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

def load_classifier(use_smart_model=True):

    if use_smart_model:
        MODEL = smart_model
    else:
        MODEL = dumb_model

    classifier = pipeline("zero-shot-classification", model=MODEL)
    return classifier

def politics_filter(articles: List[Dict]) -> List[Dict]:

    classifier = load_classifier(False)
    LABELS = ["Other", "Magyar Politika"]

    texts_to_filter = [f"{article['title']}. {article['description']}" for article in articles]

    results = classifier(texts_to_filter, LABELS)
    return results 
