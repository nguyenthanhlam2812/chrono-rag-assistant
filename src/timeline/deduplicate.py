import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_events(
    events: List[Dict[str, Any]],
    similarity_threshold: float = 0.78,
    year_diff_threshold: int = 1,
    model_name: str = 'all-MiniLM-L6-v2'
) -> List[Dict[str, Any]]:
    """
    Group similar event sentences into clusters and select representatives.
    
    Each event dict has at minimum: sentence_id, text, extracted_year, event_prob, event_type, doc_id, topic
    """
    if not events:
        return []

    # 1. Encode all event texts
    texts = [e.get("text", "") for e in events]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)

    # 2. Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # 3. Cluster events greedily
    n_events = len(events)
    visited = set()
    clusters = []

    for i in range(n_events):
        if i in visited:
            continue
            
        cluster_indices = [i]
        visited.add(i)
        
        year_i = events[i].get("extracted_year")
        
        for j in range(i + 1, n_events):
            if j in visited:
                continue
                
            sim = sim_matrix[i, j]
            year_j = events[j].get("extracted_year")
            
            # Check similarity threshold
            if sim >= similarity_threshold:
                # Check year difference if both have years
                if year_i is not None and year_j is not None:
                    if abs(year_i - year_j) <= year_diff_threshold:
                        cluster_indices.append(j)
                        visited.add(j)
                else:
                    # If one or both lack years, group based on high similarity
                    # We can use a slightly higher threshold or just group them
                    if sim >= similarity_threshold + 0.05:
                        cluster_indices.append(j)
                        visited.add(j)
                        
        clusters.append(cluster_indices)

    # 4. Select representative from each cluster
    representatives = []
    
    for cluster_indices in clusters:
        cluster_events = [events[idx] for idx in cluster_indices]
        
        # Sort cluster events by priority:
        # 1. highest event_prob
        # 2. highest date_confidence
        # 3. shortest text
        cluster_events.sort(key=lambda e: (
            e.get("event_prob", 0.0),
            e.get("date_confidence", 0.0),
            -len(e.get("text", ""))
        ), reverse=True)
        
        representative = cluster_events[0].copy()
        representative["cluster_size"] = len(cluster_events)
        
        # Collect all sources in the cluster
        sources = []
        seen_docs = set()
        for e in cluster_events:
            doc_id = e.get("doc_id")
            if doc_id and doc_id not in seen_docs:
                sources.append({
                    "doc_id": doc_id,
                    "title": e.get("title", ""),  # Title might not be present in sentence, so fallback
                    "source_url": e.get("source_url", ""),
                    "chunk_id": e.get("chunk_id", "")
                })
                seen_docs.add(doc_id)
                
        representative["cluster_sources"] = sources
        representatives.append(representative)

    return representatives
