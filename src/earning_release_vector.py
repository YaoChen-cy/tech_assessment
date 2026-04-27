# Build and query a ChromaDB vector store of WM earnings call transcript chunks.
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import re

TRANSCRIPTS_DIR = "../data/raw/transcripts"
DB_LOCATION = ".././chrome_langchain_db"
COLLECTION_NAME = "earnings_transcripts"
FILENAME_RE = re.compile(r"^(\d{4})Q([1-4])\.txt$")
SPEAKER_RE = re.compile(r"^\[([^\]]+)\]\s*$")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def parse_turns(text: str):
    # Yield (speaker, content, line_number) per speaker turn, never crossing speaker boundaries.
    speaker = None
    buf: list[str] = []
    start_line = 0
    for i, line in enumerate(text.splitlines(), start=1):
        m = SPEAKER_RE.match(line.strip())
        if m:
            if speaker and any(l.strip() for l in buf):
                yield speaker, "\n".join(buf).strip(), start_line
            speaker = m.group(1).strip()
            buf = []
            start_line = i + 1
        else:
            buf.append(line)
    if speaker and any(l.strip() for l in buf):
        yield speaker, "\n".join(buf).strip(), start_line


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Sub-chunk a single speaker turn that exceeds the embedding context; chunks never cross speaker boundaries.
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


embeddings = OllamaEmbeddings(model="mxbai-embed-large")
add_documents = not os.path.exists(DB_LOCATION)

if add_documents:
    documents = []
    ids = []

    for filename in sorted(os.listdir(TRANSCRIPTS_DIR)):
        match = FILENAME_RE.match(filename)
        if not match:
            continue
        year, quarter = match.group(1), match.group(2)
        period = f"{year}Q{quarter}"

        with open(os.path.join(TRANSCRIPTS_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()

        chunk_idx = 0
        for turn_idx, (speaker, content, line_start) in enumerate(parse_turns(text)):
            for sub in split_text(content):
                doc_id = f"{period}-{chunk_idx}"
                documents.append(Document(
                    page_content=sub,
                    metadata={
                        "fiscal_year": int(year),
                        "quarter": int(quarter),
                        "period": period,
                        "source": filename,
                        "speaker": speaker,
                        "line_start": line_start,
                        "turn_index": turn_idx,
                    },
                    id=doc_id,
                ))
                ids.append(doc_id)
                chunk_idx += 1

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

def _build_filter(period: str | None, year: int | None, quarter: int | None):
    # Build a Chroma metadata filter dict from optional period, year, and quarter arguments.
    filters = {}
    if period:
        filters["period"] = period
    if year is not None:
        filters["fiscal_year"] = year
    if quarter is not None:
        filters["quarter"] = quarter
    if not filters:
        return None
    if len(filters) == 1:
        return filters
    return {"$and": [{k_: v} for k_, v in filters.items()]}


def get_retriever(period: str | None = None, year: int | None = None, quarter: int | None = None, k: int = 5):
    # Return a retriever optionally scoped to a fiscal period, year, or quarter.
    search_kwargs: dict = {"k": k}
    chroma_filter = _build_filter(period, year, quarter)
    if chroma_filter is not None:
        search_kwargs["filter"] = chroma_filter
    return vector_store.as_retriever(search_kwargs=search_kwargs)


retriever = get_retriever(k=2)


if __name__ == "__main__":
    user_period = input("Enter fiscal period (e.g., 2023Q2), or leave blank for all: ").strip() or None
    user_question = input("Ask a question: ").strip()
    for doc in get_retriever(period=user_period).invoke(user_question):
        print(f"[{doc.metadata['period']}] {doc.page_content[:300]}...\n")
