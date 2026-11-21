import logging  
from pathlib import Path  

from langchain_community.vectorstores import FAISS 
from langchain_community.vectorstores.utils import DistanceStrategy  
from langchain_core.documents import Document  
from langchain_core.embeddings import Embeddings  

from src.config import INDEX_DIR 

logger = logging.getLogger(__name__)  


DISTANCE = DistanceStrategy.MAX_INNER_PRODUCT


def build_index(
    chunks: list[Document],
    embeddings: Embeddings,
    progress=None,        
    batch_size: int = 64,  
) -> FAISS:
    if not chunks: 
        raise ValueError("Geen chunks om te indexeren")

    texts = [chunk.page_content for chunk in chunks]  
    metadatas = [chunk.metadata for chunk in chunks]  

    vectors: list[list[float]] = []  
    for start in range(0, len(texts), batch_size):  
        batch = texts[start:start + batch_size]  
        vectors.extend(embeddings.embed_documents(batch))  
        if progress is not None:  
            progress(min(start + batch_size, len(texts)) / len(texts)) 

    store = FAISS.from_embeddings(
        list(zip(texts, vectors)),  
        embeddings,                 
        metadatas=metadatas,
        distance_strategy=DISTANCE,
    )
    logger.info("Index gebouwd met %d vectoren", store.index.ntotal) 
    return store


def save_index(store: FAISS, directory: str | Path = INDEX_DIR) -> None:  
    """Slaat de index op schijf op."""
    directory = Path(directory)  
    directory.mkdir(parents=True, exist_ok=True)  
    store.save_local(str(directory))  
    logger.info("Index opgeslagen in %s", directory)


def index_exists(directory: str | Path = INDEX_DIR) -> bool: 
    return (Path(directory) / "index.faiss").exists() 


def load_index(embeddings: Embeddings, directory: str | Path = INDEX_DIR) -> FAISS:
    """Laadt een eerder opgeslagen index."""
    directory = Path(directory)
    if not index_exists(directory):  
        raise FileNotFoundError(f"Geen index gevonden in {directory}")
    
    return FAISS.load_local(
        str(directory),
        embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy=DISTANCE,
    )