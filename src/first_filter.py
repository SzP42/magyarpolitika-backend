import torch
from transformers import pipeline
from typing import List, Dict

smart_model = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
dumb_model = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

def load_classifier(use_smart_model=True):

    device = -1
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"

    if use_smart_model:
        MODEL = smart_model
        print(f"[Filter] Loading smart model: {MODEL}")
    else:
        MODEL = dumb_model
        print(f"[Filter] Loading fast model: {MODEL}")

    classifier = pipeline("zero-shot-classification", model=MODEL, device=device)
    print(f"[Filter] Model loaded successfully")
    return classifier

def politics_filter(articles: List[Dict], use_smart_model: bool = False) -> List[Dict]:

    print(f"[Filter] Starting politics filter on {len(articles)} articles...")
    classifier = load_classifier(use_smart_model)
    LABELS = ["Other", "Hungarian Politics"]

    texts_to_filter = [f"{article['title']}. {article['description']}" for article in articles]

    # results = classifier(texts_to_filter, LABELS)

    print(f"[Filter] Classifying articles...")
    results = [article for article in articles if classifier(f"{article['title']}. {article['description']}", LABELS)["labels"][0] == "Hungarian Politics"]

    print(f"[Filter] Filtered {len(articles) - len(results)} articles out of {len(articles)}")
    print(f"[Filter] {len(results)} articles passed the politics filter")

    return results 


