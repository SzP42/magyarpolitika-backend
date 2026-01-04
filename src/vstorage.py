import os 
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import Dict, List
import numpy as np

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index = pc.Index(host="https://politics-app-551uxi4.svc.aped-4627-b74a.pinecone.io")

# A function to store the new articles in the vector db

# A function to get the articles from the topic memory
def query_topic_memory():

    # 1. We query the namespace. 
    # Since we want to pull 'all' topics (up to a limit), we can 
    # use a zero-vector if we just want to fetch based on existence.
    # Note: top_k should be larger than your expected number of topics.
    results = index.query(
        namespace="topic_memory",
        vector=[0.0] * 384, 
        top_k=1000, 
        include_metadata=True,
        include_values=True
    )

    topic_memory_dict = {}

    for match in results['matches']:
        original_name = match['metadata']['original_name']
        vector = np.array(match['values'])

        if original_name:
            topic_memory_dict[original_name] = vector

    print(f"[Storage] Fetched {len(topic_memory_dict)} topics from memory.")
    return topic_memory_dict # Dict[original_name(str): vector(np.array)]


def upsert_to_namespace(topic_name, slug_id, articles, save_to_memory=False):
    if save_to_memory:
        print("save to memory on")
        topic_vectors = list(articles[0]['vector'])
        # Convert numpy array to list if needed
            
        print(f"[Storage] Saving topic '{topic_name}' to topic_memory with slug '{slug_id}'")
        vector_to_upsert = [{
            "id": slug_id,
            "values": topic_vectors,
            "metadata": {
                "original_name": topic_name
            }
        }]
        result = index.upsert(vectors=vector_to_upsert, namespace="topic_memory")
        print(f"[Storage] upserted {topic_name} to topic memory, result: {result}")

    vectors_to_upsert = []

    for article in articles:
        # print(f"[Storage] Upserting {article['title']} to {slug_id} namespace")

        vectors_to_upsert.append({
            "id": article['link'], 
            "values": article['vector'],
            "metadata": {
                "title": article['title'],
                "description": article['description'],
                "published": article['published'],
                "url": article['link'],
                "source": article['source'],
                }
            })

    # Upsert all articles at once, outside the loop
    if vectors_to_upsert:
        try:
            result = index.upsert(vectors=vectors_to_upsert, namespace=slug_id)
            # print(f"[Storage] Successfully upserted {len(vectors_to_upsert)} articles to {slug_id} namespace. Upserted count: {result.get('upserted_count', 'unknown')}")
        except Exception as e:
            print(f"[Storage] ERROR: Failed to upsert articles to {slug_id} namespace: {e}")
            raise



