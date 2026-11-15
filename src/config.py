#General configuration of the RAG-bot
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
INDEX_DIR = ROOT_DIR / "data" / "index"

LLM_MODEL = "mistral"
EMBEDDING_MODEL = "infloat/multilingual-e5-small"