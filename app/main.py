import logging 
import sys  
import time  
from datetime import datetime 
from pathlib import Path  

ROOT = Path(__file__).resolve().parent.parent 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  

from src.advanced import ConversationHistory  
from src.config import EMBEDDING_MODEL, INDEX_DIR, LLM_MODEL, LLM_NUM_CTX, LLM_NUM_PREDICT, MULTI_QUERY_ENABLED, RAW_DIR, RETRIEVAL_K
from src.embeddings import E5Embeddings 
from src.generation import create_llm, generate_stream  
from src.indexing import clear_index, delete_pdfs, get_status, list_pdfs, rebuild_index, save_uploads  
from src.pipeline import format_sources, prepare_request  
from src.vectorstore import load_index  
logger = logging.getLogger(__name__)


st.set_page_config(page_title="Documentassistent", layout="centered", initial_sidebar_state="expanded")

AVATARS = {"user": ":material/person:", "assistant": ":material/menu_book:"}  




@st.cache_resource(show_spinner="Embeddingmodel laden...")  
def get_embeddings():
    return E5Embeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return create_llm()  #


@st.cache_resource(show_spinner="Index laden...")
def load_store(index_version: int):  
    return load_index(get_embeddings(), INDEX_DIR)


def index_version() -> int:
    return (INDEX_DIR / "index.faiss").stat().st_mtime_ns  


def format_time(iso_text) -> str:
    try:
        return datetime.fromisoformat(iso_text).strftime("%d-%m-%Y om %H:%M")
    except (TypeError, ValueError):
        return "een onbekend tijdstip"



def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  
    if "history" not in st.session_state:
        st.session_state.history = ConversationHistory()  
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0  


def show_notice() -> None:
    notice = st.session_state.pop("notice", None)  
    if notice:
        kind, text = notice 
        getattr(st, kind)(text) 


def run_rebuild(intro: list[str]) -> None:
    report = None
    with st.status("Index opbouwen...", expanded=True) as box:
        bar = st.progress(0.0, text="Starten")

        def progress(fraction: float, text: str) -> None:  
            bar.progress(min(max(fraction, 0.0), 1.0), text=text)

        try:
            report = rebuild_index(get_embeddings(), RAW_DIR, INDEX_DIR, progress)
        except Exception as error: 
            logger.exception("Index opbouwen mislukt")
            box.update(label="Opbouwen mislukt", state="error")
            st.session_state.notice = ("error", "\n\n".join(intro + [f"De index kon niet worden opgebouwd: {error}"]))

    if report is not None:
        load_store.clear() 
        parts = intro + [f"Index opgebouwd: {report.vectors} chunks uit {len(report.files)} document(en) in {report.seconds:.0f} s."]
        parts += [f"Waarschuwing: {warning}" for warning in report.warnings]
        st.session_state.notice = ("warning" if report.warnings else "success", "\n\n".join(parts))
    st.session_state.reset_counter += 1  
    st.rerun()  


def process_uploads(uploaded_files) -> None:
    files = [(file.name, file.getvalue()) for file in uploaded_files]  
    saved, replaced, rejected = save_uploads(files, RAW_DIR)
    intro = []
    if saved:
        intro.append(f"Toegevoegd: {', '.join(sorted(set(saved)))}.")
    if replaced:
        intro.append(f"Bestaande bestanden met dezelfde naam zijn overschreven: {', '.join(sorted(set(replaced)))}.")
    if rejected:
        intro.append("Niet toegevoegd:\n" + "\n".join(f"- {reason}" for reason in rejected))
    if not saved: 
        st.session_state.notice = ("error", "\n\n".join(intro))
        st.session_state.reset_counter += 1
        st.rerun()
    run_rebuild(intro)  


def run_delete(names: list[str]) -> None:
    removed = delete_pdfs(names, RAW_DIR)
    intro = [f"Verwijderd: {', '.join(removed)}."] if removed else []
    if list_pdfs(RAW_DIR):  
        run_rebuild(intro)
    clear_index(INDEX_DIR) 
    load_store.clear()
    st.session_state.notice = ("info", "\n\n".join(intro + ["Er zijn geen documenten meer. De index is gewist."]))
    st.session_state.reset_counter += 1
    st.rerun()


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.history.clear()


def describe_status(status) -> None:
    if status.state == "none":
        st.info("Er is nog geen kennisbasis.")
    elif status.state == "current":
        st.success("De index is actueel.")
    elif status.state == "unknown":
        st.warning("Er is een index, maar zonder bouwgegevens (waarschijnlijk via de commandoregel gemaakt). Bouw opnieuw op om de status bij te houden.")
    else:
        st.warning("De index is verouderd:\n" + "\n".join(f"- {reason}" for reason in status.reasons))
        if not status.usable:
            st.error("De index werd met een ander embeddingmodel gebouwd en kan niet gebruikt worden.")


def describe_documents(status) -> None:
    manifest = status.manifest
    if manifest: 
        for name, info in manifest["files"].items():
            st.markdown(f"**{name}**")
            st.caption(f"{info['pages']} pagina's met tekst, {info['chunks']} chunks")
        st.caption(f"{manifest['vectors']} chunks in de index, opgebouwd op {format_time(manifest.get('built_at'))}.")
    elif status.pdf_names:  
        st.markdown("\n".join(f"- {name}" for name in status.pdf_names))


def render_sidebar(status) -> None:
    version = st.session_state.reset_counter 
    with st.sidebar:
        st.header("Kennisbasis")
        describe_status(status)
        describe_documents(status)
        if status.state != "none":
            needs_rebuild = status.state in ("stale", "unknown") 
            if st.button(
                "Index opnieuw opbouwen",
                type="primary" if needs_rebuild else "secondary",
                key="rebuild_button",
                help="Leest alle PDF's opnieuw in, maakt nieuwe chunks en embeddings en overschrijft de bestaande index.",
            ):
                run_rebuild([])

        st.subheader("Documenten toevoegen")
        uploaded = st.file_uploader(
            "PDF-bestanden", type=["pdf"], accept_multiple_files=True,
            key=f"uploader_{version}", label_visibility="collapsed",
        )
        if st.button("Verwerk documenten", type="primary", disabled=not uploaded, key="process_button"):
            process_uploads(uploaded)
        st.caption("De volledige index wordt opnieuw opgebouwd en overschreven. Bestaande documenten blijven behouden.")

        st.header("Instellingen")
        st.toggle("Gespreksgeschiedenis", value=True, key="use_history",
                  help="Herschrijft vervolgvragen tot zelfstandige vragen. Kost een extra modelaanroep.")
        st.toggle("Multi-Query Retrieval", value=MULTI_QUERY_ENABLED, key="use_multi_query",
                  help="Zoekt met meerdere formuleringen van de vraag. Kost een extra modelaanroep.")
        st.slider("Aantal chunks (k)", min_value=1, max_value=10, value=RETRIEVAL_K, key="k_value",
                  help="Hoeveel chunks per vraag aan het taalmodel worden gegeven.")
        if st.button("Gesprek wissen", key="clear_chat"):
            clear_chat()

        with st.expander("Documenten verwijderen"):
            if status.pdf_names:
                chosen = st.multiselect("Documenten", status.pdf_names, key=f"delete_select_{version}", label_visibility="collapsed")
                confirmed = st.checkbox("De bestanden van schijf verwijderen", key=f"delete_confirm_{version}")
                if st.button("Verwijderen", disabled=not (chosen and confirmed), key="delete_button"):
                    run_delete(chosen)
            else:
                st.caption("Er zijn geen documenten.")

        st.divider()
        st.caption(f"Taalmodel: {LLM_MODEL} (Ollama). Embeddings: {EMBEDDING_MODEL.split('/')[-1]}.")


def render_extras(message: dict) -> None:
    sources = message.get("sources")
    if sources:
        with st.expander(f"Bronnen ({len(sources)})"): 
            for source in sources:
                st.markdown(f"**{source['label']}**")
                excerpt = source["text"]
                st.text(excerpt[:500] + ("..." if len(excerpt) > 500 else "")) 

    details = message.get("details")
    if details:
        with st.expander("Details"):
            if details["standalone"] != details["question"]:
                st.text("Herformuleerd tot: " + details["standalone"])
            if len(details["queries"]) > 1:
                st.text("Zoekvragen:\n" + "\n".join(f"- {query}" for query in details["queries"]))
            tokens = f"Prompt: ongeveer {details['prompt_tokens']} tokens"
            if details["dropped"]:
                tokens += f", {details['dropped']} chunk(s) weggelaten om te passen"
            st.text(tokens)
            st.text(f"Voorbereiding: {details['prep_seconds']:.1f} s | Antwoord: {details['answer_seconds']:.1f} s")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
            render_extras(message)


def explain_error(error: Exception) -> str:
    text = str(error)
    if any(word in text.lower() for word in ("connect", "refused", "10061")):
        return "Het taalmodel is niet bereikbaar. Start Ollama en controleer of het model beschikbaar is (commando: ollama list)."
    return f"Er ging iets mis bij het beantwoorden: {text}"


def handle_question(question: str) -> None:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            llm = get_llm()
            store = load_store(index_version())
            history = st.session_state.history
            started = time.perf_counter()
            with st.spinner("Relevante chunks zoeken..."):
                result = prepare_request(
                    question,
                    store,
                    k=st.session_state.k_value,
                    num_ctx=getattr(llm, "num_ctx", None) or LLM_NUM_CTX,  
                    reserve=getattr(llm, "num_predict", None) or LLM_NUM_PREDICT,
                    llm=llm,
                    history=history if st.session_state.use_history else None,
                    multi_query=st.session_state.use_multi_query,
                )
            prep_seconds = time.perf_counter() - started

            started = time.perf_counter()
            answer = st.write_stream(generate_stream(llm, result.messages))  
            answer_seconds = time.perf_counter() - started
        except Exception as error:
            logger.exception("Fout bij het beantwoorden van de vraag")
            message = {"role": "assistant", "content": explain_error(error), "error": True}
            st.session_state.messages.append(message)
            st.error(message["content"])
            return  

        message = {
            "role": "assistant",
            "content": answer,
            "sources": [
                {"label": label, "text": source.document.page_content}
                for label, source in zip(format_sources(result.sources), result.sources)
            ],
            "details": {
                "question": result.question,
                "standalone": result.standalone_question,
                "queries": result.queries,
                "prompt_tokens": result.prompt_tokens,
                "dropped": result.dropped,
                "prep_seconds": prep_seconds,
                "answer_seconds": answer_seconds,
            },
        }
        render_extras(message)  
        st.session_state.messages.append(message)
        history.add(question, answer) 



init_state()
status = get_status(RAW_DIR, INDEX_DIR)  
render_sidebar(status)

st.title("Documentassistent")
st.caption("Stel vragen over de documenten in de kennisbasis. Elk antwoord verwijst naar bestand en pagina.")
show_notice()

ready = status.usable 
if status.state == "none":
    st.info("De kennisbasis is leeg. Voeg PDF's toe in de zijbalk en klik op 'Verwerk documenten'.")
elif not ready:
    st.error("De index kan niet gebruikt worden. Bouw hem opnieuw op in de zijbalk.")
elif status.state in ("stale", "unknown"):
    st.warning("De index komt mogelijk niet overeen met de huidige documenten of instellingen. Zie de zijbalk.")

for message in st.session_state.messages:  
    render_message(message)

if question := st.chat_input("Stel een vraag over de documenten", disabled=not ready):
    handle_question(question)