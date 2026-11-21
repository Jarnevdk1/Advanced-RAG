import logging 
from dataclasses import dataclass, field 

from langchain_community.vectorstores import FAISS  

from src.advanced import ConversationHistory, condense_question, generate_queries
from src.config import LLM_NUM_CTX, LLM_NUM_PREDICT, MULTI_QUERY_ENABLED, MULTI_QUERY_VARIANTS, RETRIEVAL_K
from src.generation import build_prompt, estimate_prompt_tokens, fits_in_context, generate, page_label
from src.retrieval import RetrievedChunk, retrieve_multi

logger = logging.getLogger(__name__)


#Dataclass which contains everything belonging to a single prompt
@dataclass
class RAGResult:
    question: str                  #the question
    sources: list[RetrievedChunk]  #the retrieved chunks
    messages: list                 #the complete prompt as the LLM receives it
    prompt_tokens: int             #enstimated amount of tokens of the prompt
    fits: bool                     #does prompt fit inside context window
    dropped: int = 0               #amount of chunks left out to fit inside context window
    answer: str = ""               #answer of the model (starts empty)
    standalone_question: str = ""  #question after refomulation
    queries: list[str] = field(default_factory=list) #questions used (MQR)


#Retrieve + format + verify context window
#Doesn't call the LLM yet
def prepare_request(
    question: str,
    store: FAISS,
    k: int = RETRIEVAL_K,
    num_ctx: int = LLM_NUM_CTX,
    reserve: int = LLM_NUM_PREDICT,
    enforce_context: bool = True,  #False = always send prompt, even if doesnt fit in context window
    llm=None,
    history: ConversationHistory | None = None,
    multi_query: bool = MULTI_QUERY_ENABLED,    
    n_variants: int = MULTI_QUERY_VARIANTS,
) -> RAGResult:

    question = question.strip()  
    if not question:  
        raise ValueError("De vraag is leeg")

    #Step 0: Turn follow-up question into standalone question
    standalone = condense_question(llm, question, history) if (llm is not None and history) else question

    #Step 1: Retrieve relevant chunks (top k) (MRQ is possible)
    queries = generate_queries(llm, standalone, n_variants) if (multi_query and llm is not None) else [standalone]
    sources = retrieve_multi(store, queries, k)

    #Step 2: Build the entire prompt
    messages = build_prompt(question, [source.document for source in sources]) 

    #Step 3: Drop least relevant chunks if doesnt fit in context window
    dropped = 0  
    if enforce_context:
        
        while len(sources) > 1 and not fits_in_context(messages, num_ctx, reserve):
            sources.pop()  
            dropped += 1
            messages = build_prompt(question, [source.document for source in sources]) 
        if dropped:
            logger.warning("%d chunk(s) weggelaten om in het contextvenster (%d tokens) te passen", dropped, num_ctx)

    fits = fits_in_context(messages, num_ctx, reserve)  
    if not fits:  
        logger.warning("De prompt past mogelijk niet in het contextvenster (%d tokens): het begin kan worden afgekapt", num_ctx)

    return RAGResult(
        question=question,
        sources=sources,
        messages=messages,
        prompt_tokens=estimate_prompt_tokens(messages),
        fits=fits,
        dropped=dropped,
        standalone_question=standalone,
        queries=queries
    )


#The complete RAG-pipline without streaming: Question goes in, answer and sources come out
def ask(
    question: str,
    store: FAISS,
    llm,
    k: int = RETRIEVAL_K,
    enforce_context: bool = True,
    history: ConversationHistory | None = None,
    multi_query: bool = MULTI_QUERY_ENABLED,
    n_variants: int = MULTI_QUERY_VARIANTS,
) -> RAGResult:

    num_ctx = getattr(llm, "num_ctx", None) or LLM_NUM_CTX
    reserve = getattr(llm, "num_predict", None) or LLM_NUM_PREDICT
    result = prepare_request(question, store, k, num_ctx, reserve, enforce_context, llm=llm, history=history, multi_query=multi_query, n_variants=n_variants) 
    result.answer = generate(llm, result.messages) 
    if history is not None:
        history.add(result.question, result.answer)
    return result



def format_sources(sources: list[RetrievedChunk]) -> list[str]:
    return [
        f"[{number}] {source.document.metadata.get('source')}, pagina {page_label(source.document.metadata)} (score {source.score:.3f})"
        for number, source in enumerate(sources, start=1) 
    ]