import logging  
from pathlib import Path  

from langchain_community.vectorstores import FAISS 
from langchain_community.vectorstores.utils import DistanceStrategy  
from langchain_core.documents import Document  
from langchain_core.embeddings import Embeddings 

from src.config import INDEX_DIR  

logger = logging.getLogger(__name__)  

#The distance metric we use to determine the best chunks for retrieval
#Inner product with normalized vectors = cosine similarity
#The closer to score to 1, the higher the similarity
DISTANCE = DistanceStrategy.MAX_INNER_PRODUCT


#Takes our chunks and embeddings -> stores it in an FAISS index store
def build_index(chunks: list[Document], embeddings: Embeddings) -> FAISS:  # chunks + model erin, index eruit
    if not chunks:  
        raise ValueError("Geen chunks om te indexeren") 
    
    #from_documents internally calls the embedding.embed_documents and stores vectors + chunks
    store = FAISS.from_documents(chunks, embeddings, distance_strategy=DISTANCE)

    #show total amount of stored vectors
    logger.info("Index gebouwd met %d vectoren", store.index.ntotal)  
    return store



def save_index(store: FAISS, directory: str | Path = INDEX_DIR) -> None:  
    directory = Path(directory)  
    directory.mkdir(parents=True, exist_ok=True)  
    store.save_local(str(directory)) 
    logger.info("Index opgeslagen in %s", directory)



#Check if vector store already exits -> don't need to remake it if it already exists!
def index_exists(directory: str | Path = INDEX_DIR) -> bool:  
    return (Path(directory) / "index.faiss").exists() 


#Load previously saved vector store
def load_index(embeddings: Embeddings, directory: str | Path = INDEX_DIR) -> FAISS:
    directory = Path(directory)
    if not index_exists(directory):  
        raise FileNotFoundError(f"Geen index gevonden in {directory}")

    
    #allow_dangerous_deserialization: index.pkl is a pickle-file
    return FAISS.load_local(
        str(directory),
        embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy=DISTANCE,
    )