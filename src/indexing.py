import json 
import logging 
import os  
import tempfile  
import time  
from dataclasses import dataclass, field 
from datetime import datetime 
from pathlib import Path  

import pymupdf 

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, INDEX_DIR, MIN_CHUNK_CHARS, RAW_DIR 
from src.ingestion import chunk_documents, load_all_pdfs  
from src.vectorstore import build_index, index_exists, save_index  

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"  

SETTING_LABELS = {
    "chunk_size": "Chunkgrootte",
    "chunk_overlap": "Chunkoverlap",
    "min_chunk_chars": "Minimale chunklengte",
    "embedding_model": "Embeddingmodel",
}


@dataclass
class IndexStatus:
    state: str                   
    reasons: list[str]            
    usable: bool                 
    manifest: dict | None         
    pdf_names: list[str] = field(default_factory=list)  


@dataclass
class BuildReport:
    files: dict[str, dict]  
    vectors: int           
    warnings: list[str]     
    seconds: float          


def list_pdfs(pdf_dir: str | Path = RAW_DIR) -> list[Path]:
    return sorted(Path(pdf_dir).glob("*.pdf"))  


def current_settings() -> dict:
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "embedding_model": EMBEDDING_MODEL,
    }


def read_manifest(index_dir: str | Path = INDEX_DIR) -> dict | None:
    path = Path(index_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")) 
    except (json.JSONDecodeError, OSError): 
        logger.warning("Manifest kon niet gelezen worden: %s", path)
        return None


def get_status(pdf_dir: str | Path = RAW_DIR, index_dir: str | Path = INDEX_DIR) -> IndexStatus:
    pdf_dir, index_dir = Path(pdf_dir), Path(index_dir)
    pdf_names = [path.name for path in list_pdfs(pdf_dir)]

    if not index_exists(index_dir):  
        return IndexStatus("none", [], False, None, pdf_names)

    manifest = read_manifest(index_dir)
    if manifest is None:  
        return IndexStatus("unknown", [], True, None, pdf_names)

    reasons: list[str] = []
    usable = True
    built_with = manifest.get("settings", {})  
    now = current_settings() 
    for key, label in SETTING_LABELS.items():
        if built_with.get(key) != now[key]:
            reasons.append(f"{label} is gewijzigd (index: {built_with.get(key)}, nu: {now[key]})")
            if key == "embedding_model": 
                usable = False

    indexed = manifest.get("files", {})  
    for name in pdf_names:
        if name not in indexed:
            reasons.append(f"Nieuw document, nog niet in de index: {name}")
        elif indexed[name].get("size") != (pdf_dir / name).stat().st_size:  
            reasons.append(f"Document gewijzigd sinds het opbouwen: {name}")
    for name in indexed:
        if name not in pdf_names:
            reasons.append(f"Document ontbreekt op schijf, maar zit nog in de index: {name}")

    return IndexStatus("stale" if reasons else "current", reasons, usable, manifest, pdf_names)


def validate_pdf(data: bytes) -> str | None:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf: 
            if pdf.needs_pass:
                return "met een wachtwoord beveiligd"
            if len(pdf) == 0:
                return "bevat geen pagina's"
    except Exception: 
        return "geen geldige PDF"
    return None


def save_uploads(files: list[tuple[str, bytes]], pdf_dir: str | Path = RAW_DIR):
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    replaced: list[str] = []
    rejected: list[str] = []
    for raw_name, data in files:
        stem = Path(raw_name).stem  
        if not raw_name.lower().endswith(".pdf") or not stem:
            rejected.append(f"{Path(raw_name).name}: geen PDF-bestand")
            continue
        problem = validate_pdf(data)
        if problem:
            rejected.append(f"{Path(raw_name).name}: {problem}")
            continue
        target = pdf_dir / f"{stem}.pdf"  
        if target.exists():
            replaced.append(target.name)  
        target.write_bytes(data)
        saved.append(target.name)
    return saved, replaced, rejected


def delete_pdfs(names: list[str], pdf_dir: str | Path = RAW_DIR) -> list[str]:
    pdf_dir = Path(pdf_dir)
    removed = []
    for raw_name in names:
        target = pdf_dir / Path(raw_name).name  
        if target.suffix.lower() == ".pdf" and target.is_file():
            target.unlink()
            removed.append(target.name)
    return removed


def clear_index(index_dir: str | Path = INDEX_DIR) -> None:
    for name in ("index.faiss", "index.pkl", MANIFEST_NAME):
        (Path(index_dir) / name).unlink(missing_ok=True) 


def _file_stats(pdf_paths: list[Path], documents, chunks) -> dict[str, dict]:
    stats = {}
    for path in pdf_paths:
        stats[path.name] = {
            "size": path.stat().st_size,
            "pages": sum(1 for doc in documents if doc.metadata["source"] == path.name),
            "chunks": sum(1 for chunk in chunks if chunk.metadata["source"] == path.name),
        }
    return stats


def rebuild_index(
    embeddings,
    pdf_dir: str | Path = RAW_DIR,
    index_dir: str | Path = INDEX_DIR,
    progress=None,
) -> BuildReport:
    
    pdf_dir, index_dir = Path(pdf_dir), Path(index_dir)
    started = time.perf_counter()

    def report(fraction: float, text: str) -> None: 
        if progress is not None:
            progress(fraction, text)

    pdf_paths = list_pdfs(pdf_dir)
    if not pdf_paths:
        raise ValueError("Er zijn geen PDF's om te verwerken.")

    report(0.02, "PDF's inlezen")
    documents, warnings = load_all_pdfs(pdf_dir)  
    if not documents:
        raise ValueError("Geen uitleesbare tekst gevonden in de PDF's. " + " ".join(warnings))

    report(0.10, "Tekst opsplitsen in chunks")
    chunks = chunk_documents(  
        documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, min_chunk_chars=MIN_CHUNK_CHARS
    )

    report(0.15, f"Embeddings berekenen voor {len(chunks)} chunks")
    store = build_index(  
        chunks,
        embeddings,
        progress=lambda fraction: report(0.15 + 0.75 * fraction, f"Embeddings berekenen ({int(fraction * 100)}%)"),
    )

    report(0.92, "Index opslaan")
    files = _file_stats(pdf_paths, documents, chunks)
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "settings": current_settings(),
        "files": files,
        "vectors": store.index.ntotal,
    }

    index_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=index_dir) as tmp:  
        save_index(store, tmp)
        (Path(tmp) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        for name in ("index.faiss", "index.pkl", MANIFEST_NAME):  
            os.replace(Path(tmp) / name, index_dir / name)  

    report(1.0, "Klaar")
    return BuildReport(files=files, vectors=store.index.ntotal, warnings=warnings, seconds=time.perf_counter() - started)