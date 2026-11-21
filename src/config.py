#General configuration of the RAG-bot
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
INDEX_DIR = ROOT_DIR / "data" / "index"

LLM_MODEL = "mistral"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

#Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_CHARS = 25

#LLM paramaters
LLM_TEMPERATURE = 0.1
LLM_NUM_CTX = 8192
LLM_NUM_PREDICT = 1024

#Amount of chunks to retrieve for the prompt
RETRIEVAL_K = 4  

#HISTORY paramaters
HISTORY_MAX_TURNS = 3       #Amount of question-answer pairs which can be used by the question condenstation
HISTORY_ANSWER_CHARS = 300  #Max amount of chars of the previous answers

#Multi-Query Retrieval paramaters
MULTI_QUERY_ENABLED = True  #Default value = MRQ On
MULTI_QUERY_VARIANTS = 3    #Amount of extra questions next to the original
RRF_K = 60                  #Reciprocal Rank Fusion 