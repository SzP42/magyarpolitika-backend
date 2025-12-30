import torch
from transformers import pipeline
from typing import List, Dict

smart_model = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
dumb_model = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

KEYWORDS = [
    "Orbán", "Szijjártó", "Rogán", "Gulyás", "Lázár", "Navracsics", "Kocsis Máté", 
    "Szentkirályi", "Deutsch", "Pintér Sándor", "Novák", "Vitályos",
    "Magyar Péter", "Gyurcsány", "Dobrev", "Karácsony", "Toroczkai", 
    "Ungár", "Márki-Zay", "Hadházy",
    "Fidesz", "KDNP", "Tisza Párt", "TISZA", "Demokratikus Koalíció", "DK", 
    "Mi Hazánk", "Momentum", "Kutyapárt", "MKKP", "Jobbik", "LMP", "MSZP",
    "választás", "kampány", "parlament", "kormány", "ellenzék", "szavazás", 
    "mandátum", "vita", "választókerület",
]

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

def politics_filter(category_titles: List[str], use_smart_model: bool = True) -> List[Dict]:

    results = []

    print(f"[Filter] Starting politics filter on {len(category_titles)} titles")
    classifier = load_classifier(use_smart_model)
    LABELS = ["Egyéb", "Politika"]

    print(f"[Filter] Classifying articles...")

    # in the future implement sorting out the non-political titles. To have data for fine-tuning
    # results = [title for title in category_titles if classifier(title, LABELS)["labels"][0] == "Hungarian Politics"]

    for title in category_titles:
        if any(k.lower() in title.lower() for k in KEYWORDS):
            results.append(title)
            continue
        elif classifier(title, LABELS)["labels"][0] == "Politika":
            results.append(title)

    print(f"[Filter] Filtered {len(category_titles) - len(results)} articles out of {len(category_titles)}")
    print(f"[Filter] {len(results)} articles passed the politics filter")

    return results 


