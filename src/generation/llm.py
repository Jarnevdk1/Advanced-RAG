from langchain_ollama import ChatOllama  

from src.config import LLM_MODEL, LLM_NUM_CTX, LLM_NUM_PREDICT, LLM_TEMPERATURE  

CHARS_PER_TOKEN = 3  


#Creates the LLM with the specified params and connects to it
def create_llm(model: str = LLM_MODEL, 
               temperature: float = LLM_TEMPERATURE, 
               num_ctx: int = LLM_NUM_CTX, 
               num_predict: int = LLM_NUM_PREDICT) -> ChatOllama:
    
    return ChatOllama(
        model=model,                
        temperature=temperature,    
        num_ctx=num_ctx,            
        num_predict=num_predict,    
    )


#Estimate the amount of tokens the prompt uses
def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1  


def estimate_prompt_tokens(messages) -> int:
    return sum(estimate_tokens(str(message.content)) for message in messages)  


def fits_in_context(messages, num_ctx: int = LLM_NUM_CTX, reserve: int = LLM_NUM_PREDICT) -> bool:
    return estimate_prompt_tokens(messages) + reserve <= num_ctx  