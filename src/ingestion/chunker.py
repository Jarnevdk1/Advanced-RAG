import logging                  
from bisect import bisect_right  

from langchain_core.documents import Document  
from langchain_text_splitters import RecursiveCharacterTextSplitter  

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS  

logger = logging.getLogger(__name__)  

PAGE_SEPARATOR = "\n\n" 


#Splits the pages of 1 source into chunks
#Each chunks holds the correct metadata
def _chunk_source(pages: list[Document], splitter: RecursiveCharacterTextSplitter, min_chunk_chars: int) -> list[Document]:
    source = pages[0].metadata["source"]  #Reads source from the first page's metadata

    parts: list[str] = []         
    offsets: list[int] = []      
    page_numbers: list[int] = []  
    cursor = 0                   
    for page in pages:
        offsets.append(cursor)                          
        page_numbers.append(page.metadata["page"])      
        parts.append(page.page_content)                 
        cursor += len(page.page_content) + len(PAGE_SEPARATOR)  
    full_text = PAGE_SEPARATOR.join(parts)  


    chunks: list[Document] = []
    for piece in splitter.create_documents([full_text]):
        text = piece.page_content  #Text of this chunk
        if len(text) < min_chunk_chars:  
            continue

        start = max(piece.metadata["start_index"], 0)  
        end = start + len(text) - 1

        page_start = page_numbers[bisect_right(offsets, start) - 1]
        page_end = page_numbers[bisect_right(offsets, end) - 1] 

        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,            
                    "page": page_start,          
                    "page_end": page_end,        
                    "chunk_index": len(chunks),  
                },
            )
        )
    return chunks


#Splits Document-objects into chunks
def chunk_documents(documents: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, min_chunk_chars: int = MIN_CHUNK_CHARS) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        #Max length in chars (not in tokens!)
        chunk_overlap=chunk_overlap,  #Overlap between consecutive chunks (for context)
        add_start_index=True,         #Store starting index of each chunk in the original text
    )

    #Keep a dictionary {key = name of source file; value = list of Document-objects (pages) belonging to that source}
    pages_per_source: dict[str, list[Document]] = {}  
    for document in documents:
        pages_per_source.setdefault(document.metadata["source"], []).append(document)

    #Create the chunks 
    chunks: list[Document] = []
    for source, pages in pages_per_source.items():
        pages.sort(key=lambda page: page.metadata["page"])  #Lambda-function: sorts the pages on page-number (just in case)
        source_chunks = _chunk_source(pages, splitter, min_chunk_chars)
        logger.info("%s: %d pagina's -> %d chunks", source, len(pages), len(source_chunks))
        chunks.extend(source_chunks)  #Combine chunks of all PDF files together
    return chunks