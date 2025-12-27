import numpy as np
from typing import List, Dict, Tuple 
from sklearn.cluster import DBSCAN
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

_embedder = None 

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder

def assign_topics(articles: List[Dict], known_topics: List[str]=None) -> Tuple[List[Dict], List[str]]:
    print(f"[Clustering] Starting topic assignment for {len(articles)} articles...")
    embedder = get_embedder()

    articles_with_categories = []

    # Vectorize the articles
    print(f"[Clustering] Embedding {len(articles)} articles...")
    texts = [f"{article['title']}: {article['description']}" for article in articles]
    article_vectors = embedder.embed_documents(texts)
    X = np.array(article_vectors)
    print(f"[Clustering] Embedding complete. Vector shape: {X.shape}")

    # DBSCAN clustering: 
    # metric='cosine': Measures angle/similarity, not distance. (With cosine, we look at the direction (topic of the article) of the vector not the length
    #   (length is higher for a 4000+ word article than for a 140 character tweet, but if they talk about the same thing it's the same direction so they'll be grouped together))
    # eps=0.4: Strictness. Lower = articles must be very similar, it's the distance within which articles will be grouped in the same cluster.
    # min_samples=3: Needs at least 3 articles within distance to form a "Cluster" (Topic).

    print(f"[Clustering] Running DBSCAN clustering (eps=0.4, min_samples=3)...")
    clustering = DBSCAN(metric='cosine', eps=0.4, min_samples=3).fit(X)
    # each cluster is assigned a label (0, 1, 2, ...) which is the index of the cluster, 
    # -1 is for noise (articles that don't fit into any cluster). 
    labels = clustering.labels_
    unique_labels = set(labels)
    print(f"[Clustering] Found {len(unique_labels)} unique labels: {unique_labels}")
    # Get topics history and turn them into vectors to compare them with the new articles. 
    if known_topics is None:
        known_topics = []
        known_topic_vectors = None
        print(f"[Clustering] No known topics provided, all clusters will be new topics")
    elif len(known_topics) > 0:
        print(f"[Clustering] Comparing against {len(known_topics)} known topics...")
        known_topic_vectors = np.array(embedder.embed_documents(known_topics))
    else:
        known_topic_vectors = None
        print(f"[Clustering] No known topics provided, all clusters will be new topics")

    new_topics = {} # Map: cluster_id --> final_topic_name

    # Process clusters, make a list of all new articles about the topic 
    for idx, label in enumerate(unique_labels):
        if label == -1:
            continue 
        print(f"[Clustering] processing label: {label} || {idx}/{len(unique_labels)-1}")
        # get all articles that belong to this cluster
        indices = [i for i, x in enumerate(labels) if x == label]
        cluster_articles = [articles[i] for i in indices]

        # propose a name for the new topic and check if it's similar to the already existing topics.
        proposed_name = cluster_articles[0]['title']
        proposed_vector = embedder.embed_query(proposed_name)

        final_topic_name = proposed_name 
        is_new_topic = True 

        if known_topic_vectors is not None and len(known_topic_vectors) > 0:
            # compare against all old topics 
            similarities = cosine_similarity([proposed_vector], known_topic_vectors)[0]

            # Find the single best match for the new proposed topic in the historical database
            best_match_id = np.argmax(similarities)
            best_score = similarities[best_match_id]

            if best_score > 0.80:
                existing_name = known_topics[best_match_id]
                final_topic_name = existing_name 
                is_new_topic = False
                print(f"[Clustering] Cluster {label} ({len(cluster_articles)} articles): Matched to existing topic '{final_topic_name}' (similarity: {best_score:.3f})")
            else:
                print(f"[Clustering] Cluster {label} ({len(cluster_articles)} articles): New topic '{final_topic_name}' (best match similarity: {best_score:.3f}, below 0.80 threshold)")
        else:
            print(f"[Clustering] Cluster {label} ({len(cluster_articles)} articles): New topic '{final_topic_name}'")

        # Update memory 
        new_topics[label] = final_topic_name

        if is_new_topic:
            known_topics.append(final_topic_name)
        if known_topic_vectors is None:
            known_topic_vectors = np.array([proposed_vector])
        else: 
            known_topic_vectors = np.vstack([known_topic_vectors, proposed_vector])

    # Assign topics to articles 
    print(f"[Clustering] Assigning topics to articles...")
    for i, article in enumerate(articles):
        cluster_id = labels[i]

        if cluster_id != -1:
            print(f"[Clustering] Article: {article}, cluster_id: {cluster_id}, topic: {new_topics[cluster_id]}")
            article['Category'] = new_topics[cluster_id]
            articles_with_categories.append(article)
            
    
    print(f"[Clustering] Topic assignment complete")

    return articles_with_categories, known_topics


