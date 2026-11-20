import re  
from dataclasses import dataclass 

from langchain_core.prompts import ChatPromptTemplate  

from src.config import HISTORY_ANSWER_CHARS, HISTORY_MAX_TURNS  
from src.generation import generate  

MAX_REWRITE_CHARS = 300  

_CITATION = re.compile(r"\s*\[\d+\]") 
_PREFIX = re.compile(
    r"^(zelfstandige vraag|herschreven vraag|standalone question|rewritten question)\s*:\s*",
    re.IGNORECASE)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Je herschrijft vervolgvragen tot zelfstandige vragen.

Regels:
1. Gebruik de gespreksgeschiedenis uitsluitend om verwijzingen op te lossen, zoals "het", "dat" of "en voor X?".
2. Is de nieuwe vraag op zichzelf al begrijpelijk, geef ze dan ongewijzigd terug.
3. Schrijf de zelfstandige vraag in exact dezelfde taal als de nieuwe vraag, ook als de geschiedenis in een andere taal is.
4. Geef alleen de vraag terug. Beantwoord ze niet. Geen uitleg, geen aanhalingstekens."""),
    ("human", """Gespreksgeschiedenis:
{history}

Nieuwe vraag: {question}

Zelfstandige vraag:"""),
])


@dataclass
class Turn: #One turn which holds the question and answer of the LLM
    question: str
    answer: str


#Stores the last few turns of the converstation
class ConversationHistory:

    def __init__(self, max_turns: int = HISTORY_MAX_TURNS):
        if max_turns < 1:  
            raise ValueError("max_turns moet minstens 1 zijn")
        self.max_turns = max_turns
        self._turns: list[Turn] = [] 

    def add(self, question: str, answer: str) -> None:
        self._turns.append(Turn(question, answer)) 
        self._turns = self._turns[-self.max_turns:]  

    def recent(self) -> list[Turn]:
        return list(self._turns)  

    def clear(self) -> None:
        self._turns = []

    def __len__(self) -> int:  
        return len(self._turns)


#Changes turns into text for the reformulation prompt
def format_history(turns: list[Turn]) -> str:
    lines = []
    for turn in turns:
        answer = _CITATION.sub("", turn.answer).strip()
        if len(answer) > HISTORY_ANSWER_CHARS:  
            answer = answer[:HISTORY_ANSWER_CHARS].rstrip() + "..."
        lines.append(f"Gebruiker: {turn.question}")
        lines.append(f"Assistent: {answer}")
    return "\n".join(lines)


#Takes the refourmulation 
def _clean_rewrite(text: str, original: str) -> str:
    for line in text.splitlines():  
        line = line.strip()
        if line:
            break
    else:  
        return original
    line = _PREFIX.sub("", line).strip().strip("\"'“”‘’").strip() 
    if not line or len(line) > MAX_REWRITE_CHARS:
        return original
    return line


#Turns a follow-up questionn into a standalone question -> if no history, the question stays unchanged
def condense_question(llm, question: str, history: ConversationHistory) -> str:
    turns = history.recent()
    if not turns: 
        return question
    messages = CONDENSE_PROMPT.format_messages(history=format_history(turns), question=question)
    return _clean_rewrite(generate(llm, messages), question)