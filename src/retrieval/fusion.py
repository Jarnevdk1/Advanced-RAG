from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document 

from src.config import RETRIEVAL_K, RRF_K  

from .retriever import RetrievedChunk, retrieve  


#Identifies a chunk -> make sure the same retrieved chunk counts as one, even when it is retrieved from multiple question with MQR
def chunk_key(document: Document) -> tuple:
    meta = document.metadata

    return (meta.get("source"), meta.get("chunk_index"), document.page_content)


#Combines ranking of multiple question of the MQR using Reciprocal Rank Fusion and returns the best k chunks
def fuse_results(result_lists: list[list[RetrievedChunk]], k: int, rrf_k: int = RRF_K) -> list[RetrievedChunk]:
    fused_scores: dict[tuple, float] = {}   
    best: dict[tuple, RetrievedChunk] = {}  

    for results in result_lists:  
        for rank, item in enumerate(results, start=1):  
            key = chunk_key(item.document)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)  
            if key not in best or item.score > best[key].score:  
                best[key] = item

    ordered = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    return [best[key] for key in ordered[:k]]


#Retrieves chunks per question 
def retrieve_multi(
    store: FAISS,
    queries: list[str],
    k: int = RETRIEVAL_K,
    k_per_query: int | None = None,  
) -> list[RetrievedChunk]:

    k_per_query = k_per_query or k
    result_lists = [retrieve(store, query, k_per_query) for query in queries]  
    return fuse_results(result_lists, k)