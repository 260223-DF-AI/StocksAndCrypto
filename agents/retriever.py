import os
from dotenv import load_dotenv

import boto3
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone
import cohere

from agents.state import ResearchState

load_dotenv()


# Module-level singletons, lazily constructed on first use. Lazy init matters:
# both clients read env vars at construction time, so eager init at import time
# would force every caller (tests, scripts, Lambda cold start) to have AWS_REGION
# and PINECONE_API_KEY set *before* `from agents.retriever import ...` runs.
_embedder = None
_pinecone_index = None
_bedrock_runtime = None


def _get_embedder():
    """Lazy-init so unit tests can monkeypatch before first call."""
    global _embedder
    if _embedder is None:
        _embedder = BedrockEmbeddings(
            model_id=os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            region_name=os.environ["AWS_REGION"],
        )
    return _embedder


def _get_index():
    """Lazy-init so unit tests can monkeypatch before first call."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    return _pinecone_index

def _get_bedrock_runtime():
    """Lazy-init so unit tests can monkeypatch before first call."""
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name = os.environ["AWS_REGION"]
        )
    return _bedrock_runtime

def _cos_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity for plain Python lists — avoids a numpy import."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _compress(chunk_text: str, query: str, max_sentences: int = 4) -> str:
    """Keep the sentences whose embedding is closest to the query.

    Cheap, deterministic, and good enough to halve token usage on long
    chunks. Replace with an LLM-based compressor if you need higher recall.
    """
    sentences = [s.strip() for s in chunk_text.split(". ") if s.strip()]
    if len(sentences) <= max_sentences:
        return chunk_text
    embedder = _get_embedder()
    query_vec = embedder.embed_query(query)
    sent_vecs = embedder.embed_documents(sentences)
    scores = [_cos_sim(query_vec, sv) for sv in sent_vecs]
    top = sorted(range(len(sentences)), key=lambda i: -scores[i])[:max_sentences]
    top.sort()                                      # preserve original order
    return ". ".join(sentences[i] for i in top)

def _rerank_matches(query: str, matches: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank Pinecone matches using Bedrock Cohere rerank."""
    print("reranking begin")
    co = cohere.Client(api_key = os.environ["COHERE_API_KEY"])
    if not matches:
        return []
    
    documents = [match.get("metadata", {}).get("content", "") for match in matches]

    rankings = co.rerank(
        model = os.environ["COHERE_RERANK_MODEL"],
        query = query,
        documents = documents,
        top_n = min(top_k, len(documents))
    )
    print("got rerank model")

    results = rankings.results
    if not results:
        return matches[:top_k]
    print("got results")
    reranked = []
    for r in results[:top_k]:
        index = r.index
        score = r.relevance_score
        if index is None or index < 0 or index >= len(matches):
            continue
        match = matches[index]
        try:
            match["rerank_score"] = float(score) if score is not None else None
        except Exception:
            match["rerank_score"] = None
        reranked.append(match)

    return reranked or matches[:top_k]

def retriever_node(state: ResearchState) -> dict:
    """Retrieve and compress."""
    plan = state.get("plan", [])
    idx = state.get("current_subtask_index", 0)
    sub_task = plan[idx] if plan else state["question"]
    log = [f"[retriever] sub-task: {sub_task!r}"]

    # 1) embed + Pinecone semantic search ------------------------------------
    index = _get_index()
    query_vec = _get_embedder().embed_query(sub_task)
    raw = index.query(
        vector=query_vec,
        top_k=5,
        namespace="primary-corpus",
        include_metadata=True,
    )
    matches = raw.get("matches", []) if isinstance(raw, dict) else raw["matches"]
    log.append(f"[retriever] pinecone returned {len(matches)} candidates")

    if not matches:
        return {"retrieved_chunks": [], "scratchpad": log + ["[retriever] no matches"]}

    # 2) rerank with Cohere --------------------------------------------------
    try:
        matches = _rerank_matches(sub_task, matches, top_k = 5)
        log.append(f"[retriever] reranked candidates with Cohere")
    except Exception as e:
        log.append(f"[retriever] reranking failed: {e}")

    # 3) compress + structure ------------------------------------------------
    # Pinecone returns matches sorted by cosine score; take top 5.
    chunks = []
    for match in matches[:5]:
        meta = match["metadata"]
        chunks.append({
            "content": _compress(meta["content"], sub_task),
            "relevance_score": float(match.get("score", 0.0)),
            "source": meta.get("source", "unknown"),
            "page_number": meta.get("page_number"),
        })
    return {"retrieved_chunks": chunks, "scratchpad": log}