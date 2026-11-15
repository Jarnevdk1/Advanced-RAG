import logging
import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


#creates a logger which can be used to check actions
#__name__ adds the name of this document to a logged action, to see where it comes from
logger = logging.getLogger(__name__)

TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES


#Basic cleaning of raw PDF-files
def clean_text(text: str) -> str:

    #Remove null-signs
    text = text.replace("\x00", "")

    #Replace 1+ spaces or tabs by a single space
    text = re.sub(r"[ \t\xa0]+", " ", text)

    #Replace line end with a space before or after by a single line end
    text = re.sub(r" ?\n ?", "\n", text)

    #Reconnect words that got seperated by a "-" on different lines
    text = re.sub(r"([^\W\d_])[-\u2010\u00ad]\n([a-z\u00e0-\u00ff])", r"\1\2", text)

    #Replace remaining "-" with a single space
    text = text.replace("\u00ad", "")

    #Replace 3+ line ends with a single line end
    text = re.sub(r"\n{3,}", "\n\n", text)

    #Strip white space before and after text
    text = text.strip()

    return text


#Loads a single PDF and stores it as a list of Documents (each Document = 1 page)
def load_pdf(path: str | Path, sort: bool = False) -> list[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF niet gevonden: {path}")

    documents = []
    with pymupdf.open(path) as pdf:
        total_pages = len(pdf) 

        #Extract text from the cleaned up PDF
        for page_number, page in enumerate(pdf, start=1):
            extracted_text = clean_text(page.get_text("text", flags=TEXT_FLAGS, sort=sort))

            #Add document to the list (= 1 page) with extracted text and metadata
            if extracted_text:
                documents.append(
                    Document(
                        page_content = extracted_text,
                        metadata = {
                            "source": path.name,
                            "page": page_number,
                            "total_pages": total_pages
                        }
                    )
                )


    #Summary per PDF
    logger.info("%s: %d van %d pagina's ingelezen", path.name, len(documents), total_pages)
    return documents


#Returns document off all PDF's combined. 
#Also returns possible warnings which can be shown in the UI
def load_all_pdfs(pdf_directory: str | Path) -> tuple[list[Document], list[str]]:
    pdf_paths = sorted(Path(pdf_directory).glob("*.pdf"))

    if not pdf_paths:
        logger.warning("Geen PDF's gevonden in %s", pdf_directory)

    documents = []
    warnings = []

    for path in pdf_paths:
        path = Path(path)
        try:
            #Try to read the PDF
            pages = load_pdf(path)
        except Exception:
            logger.exception("Fout bij het lezen van %s", path.name)
            warnings.append(f"{path.name}: kon niet gelezen worden")
            continue

        if not pages:
            warnings.append(f"{path.name}: geen uitleesbare tekst")

        documents.extend(pages)

    return documents, warnings
