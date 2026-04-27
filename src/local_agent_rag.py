# RAG-based Q&A agent over WM earnings transcripts using Ollama and LangChain.
import re
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from earning_release_vector import get_retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert analyst answering questions about Waste Management's quarterly
earnings call transcripts.

Fiscal period in scope: {period}

Each excerpt below is tagged with an opaque source id like [S1], [S2], etc.
Rules:
- When you use information from an excerpt, cite its id exactly as written,
  e.g. "...revenue guidance was raised [S2]".
- Every factual claim in your answer must end with at least one [S#] citation.
- Do NOT invent speaker names, numbers, or quotes that are not in the excerpts.
- If the excerpts don't contain the answer, say so explicitly.

Excerpts:
{excerpts}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

CITE_RE = re.compile(r"\[?\(?\b[Ss](\d+)\b\)?\]?")  # matches [S1], (S1), S1, s1, [s1]
BAD_CITE_RE = re.compile(r"\[S[^\d\]][^\]]*\]")  # [Sk], [Sn], [S?], etc.


def format_excerpts(docs):
    # Format retrieved documents as numbered excerpts for the prompt.
    return "\n\n".join(f"[S{i + 1}]\n{d.page_content}" for i, d in enumerate(docs))


def citation_for(doc):
    # Format a document's metadata into a human-readable citation string.
    m = doc.metadata
    return f"[{m['period']} | {m.get('speaker', 'Unknown')} | line {m.get('line_start', '?')}]"


def resolve_citations(answer: str, docs) -> str:
    # Replace short [S#] placeholders in the answer with full metadata citations.
    def sub(match):
        idx = int(match.group(1)) - 1
        return citation_for(docs[idx]) if 0 <= idx < len(docs) else match.group(0)
    cleaned = BAD_CITE_RE.sub("", answer)
    return CITE_RE.sub(sub, cleaned)


while True:
    print("\n\n-------------------------------")
    period = input("Fiscal period (e.g., 2023Q2, blank for all): ").strip() or None
    question = input("Ask your question (q to quit): ")
    if question == "q":
        break

    docs = get_retriever(period=period).invoke(question)
    raw = chain.invoke({
        "period": period or "all available quarters",
        "excerpts": format_excerpts(docs),
        "question": question,
    })
    print("\n")
    print(resolve_citations(raw, docs))

    cited_idxs = sorted({int(m) for m in CITE_RE.findall(raw) if 0 < int(m) <= len(docs)})
    if cited_idxs:
        print("\n" + "=" * 60)
        print("Cited excerpts:")
        print("=" * 60)
        for i in cited_idxs:
            d = docs[i - 1]
            m = d.metadata
            header = f"[S{i}] {m['period']} | {m.get('speaker')} | line {m.get('line_start')} — {m['source']}"
            print(f"\n{header}")
            print("-" * len(header))
            print(d.page_content)
    else:
        continue
