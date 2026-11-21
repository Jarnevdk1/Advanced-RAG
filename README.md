# Advanced RAG-chatbot

Lokale RAG-chatbot voor het vak Trends in Business & IT (VIVES, Toegepaste Informatica, afstudeerrichting Artificial Intelligence). De chatbot beantwoordt vragen over geüploade PDF's, met verwijzingen naar bestand en pagina, volledig lokaal.

## Functies
- PDF's uploaden en omzetten naar een FAISS-index (de index wordt bij elke wijziging volledig opnieuw opgebouwd)
- Basis-RAG: chunking, embeddings, retrieval en antwoordgeneratie
- Gespreksgeschiedenis (herformulering van vervolgvragen)
- Multi-Query Retrieval met Reciprocal Rank Fusion
- Streamlit-interface met bronnen en details per antwoord

## Technologieën
Python, LangChain, Streamlit, Ollama (Mistral 7B), `intfloat/multilingual-e5-small`, FAISS, PyMuPDF.

## Installatie
1. Installeer Python 3.11 of 3.12 en [Ollama](https://ollama.com/download).
2. Haal het taalmodel op: `ollama pull mistral`
3. Maak en activeer een virtuele omgeving en installeer de packages:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Gebruik
```
streamlit run app/main.py
```
Voeg PDF's toe in de zijbalk en klik op "Verwerk documenten". Het embeddingmodel wordt bij de eerste keer gedownload.

## Structuur
- `app/main.py`: Streamlit-interface
- `src/`: ingestie, embeddings, vectoropslag, retrieval, generatie, geavanceerde technieken en de pijplijn
- `tests/`: unittests en handmatige testscripts
- `docs/devlog.md`: ontwikkelingslogboek
- `debug_index.ipynb`: notebook om de index te verkennen

## Data
De PDF's die voor de tests werden gebruikt zijn auteursrechtelijk beschermd en staan niet in deze repository.