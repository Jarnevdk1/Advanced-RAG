from dataclasses import dataclass  

from langchain_community.vectorstores import FAISS  
from langchain_core.documents import Document  

from src.config import RETRIEVAL_K  


#Class to store a retrieved chunk + its cosine similarity score
@dataclass 
class RetrievedChunk:
    document: Document  
    score: float        


#Retrieves the k most relevent chunks (ranked from best to worst)
def retrieve(store: FAISS, question: str, k: int = RETRIEVAL_K) -> list[RetrievedChunk]:
    if k < 1: 
        raise ValueError("k moet minstens 1 zijn")
    
    results = store.similarity_search_with_score(question, k=k)
    
    return [RetrievedChunk(document=document, score=float(score)) for document, score in results]