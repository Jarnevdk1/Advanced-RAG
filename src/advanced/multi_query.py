import re  

from langchain_core.prompts import ChatPromptTemplate  

from src.config import MULTI_QUERY_VARIANTS 
from src.generation import generate 

MAX_VARIANT_CHARS = 300  

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")  

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Je helpt bij het zoeken in documenten. Je schrijft alternatieve formuleringen van een zoekvraag.

Regels:
1. Schrijf precies {n} varianten van de vraag die dezelfde informatie zoeken, 
maar andere woorden gebruiken (synoniemen, andere vaktermen, een ruimere of specifiekere formulering).
2. Schrijf elke variant in exact dezelfde taal als de oorspronkelijke vraag.
3. Voeg geen informatie toe die niet in de vraag staat.
4. Geef één variant per regel, zonder nummering en zonder uitleg."""),
    ("human", """Vraag: {question}

Varianten:"""),
])


#Takes the variant of the question from the answer of the LLM, and cleaned up
def parse_variants(text: str, original: str, n: int) -> list[str]:
    seen = {original.strip().lower()}  
    variants: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip().strip("\"'“”‘’").strip() 
        if not line or line.endswith(":") or len(line) > MAX_VARIANT_CHARS:  
            continue
        if line.lower() in seen: 
            continue
        seen.add(line.lower())
        variants.append(line)
    return variants[:n] 



def generate_queries(llm, question: str, n_variants: int = MULTI_QUERY_VARIANTS) -> list[str]:
    messages = MULTI_QUERY_PROMPT.format_messages(n=n_variants, question=question)
    variants = parse_variants(generate(llm, messages), question, n_variants)
    return [question] + variants  