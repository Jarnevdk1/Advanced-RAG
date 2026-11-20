from langchain_core.documents import Document  
from langchain_core.prompts import ChatPromptTemplate  

#System Prompt is the basic prompt for the LLM
#We can specify what it can and cannot do here
SYSTEM_PROMPT = """Je bent een assistent die vragen beantwoordt op basis van de aangeleverde documentfragmenten.

Regels:
1. Gebruik uitsluitend de informatie uit de fragmenten. Gebruik geen eigen voorkennis.
2. Staat het antwoord niet in de fragmenten, zeg dan dat het antwoord niet in de documenten staat. Verzin niets.
3. Antwoord altijd in de taal van de gebruikersvraag, ongeacht de taal van de documentfragmenten. 
Vertaal relevante informatie indien nodig, zonder de oorspronkelijke betekenis te veranderen.
4. Verwijs na elke bewering naar het fragment waarop ze steunt, bijvoorbeeld [1] of [2][3].
5. Wees bondig."""


#THE USER'S MESSAGE
#Automatically fills in the chosen chunks in the {context} and the question in the {question}
HUMAN_PROMPT = """Fragmenten:
{context}

Vraag: {question}

Antwoord in de taal van deze vraag, niet de taal van de fragmenten: "{question}\""""



#Combines the SYSTEM_PROMPT and HUMAN_PROMPT into a single PROMPT
RAG_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])


#Returns the page(s) a certain chunk was taken from
def page_label(metadata: dict) -> str:

    page = metadata.get("page")  
    page_end = metadata.get("page_end", page)  
    return str(page) if page == page_end else f"{page}-{page_end}"


#Creates numbered text blocks with source, so the model can refer to it 
def format_context(chunks: list[Document]) -> str:

    blocks = []  
    for number, chunk in enumerate(chunks, start=1):  
        source = chunk.metadata.get("source")  
        pages = page_label(chunk.metadata)  
        blocks.append(f"[{number}] (bron: {source}, pagina {pages})\n{chunk.page_content}")  
    return "\n\n".join(blocks)  


#Build the entire prompt, filling in the {context} and {question}
def build_prompt(question: str, chunks: list[Document]) -> list:
    return RAG_PROMPT.format_messages(context=format_context(chunks), question=question)