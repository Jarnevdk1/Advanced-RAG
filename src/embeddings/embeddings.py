from langchain_core.embeddings import Embeddings  
from langchain_huggingface import HuggingFaceEmbeddings 

from src.config import EMBEDDING_MODEL  

#Derives from Embeddings = we promise to have embed_documents and embed_query
#Class automatically adds prefixes to chunks/queries before embedding to a vector -> better results in retrieval later
class E5Embeddings(Embeddings):  
    
    QUERY_PREFIX = "query: "      
    PASSAGE_PREFIX = "passage: "  

    #Constructor which loads the embedding model
    #Normalizes the embeddings so it can use cosine similarity later
    def __init__(self, model_name: str = EMBEDDING_MODEL):  

        self._model = HuggingFaceEmbeddings(  
            model_name = model_name,
            encode_kwargs = {"normalize_embeddings": True},  
        )

    #Method to embed chunks (with the added prefix)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:  
        prefixed = [self.PASSAGE_PREFIX + text for text in texts]  
        return self._model.embed_documents(prefixed)  

    #Method to embed a user query (with the added prefix)
    def embed_query(self, text: str) -> list[float]:  
        return self._model.embed_query(self.QUERY_PREFIX + text)  