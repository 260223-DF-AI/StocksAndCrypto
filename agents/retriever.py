"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

from agents.state import ResearchState
from sentence_transformers import CrossEncoder
from pinecone import Pinecone
import os
from scripts.ingest import generate_embeddings

reranker = CrossEncoder("cross-encoder/msmarco-MiniLM-L-6-v2")

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=api_key)


def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve relevant document chunks for the current sub-task.

    TODO:
    - Extract the current sub-task from state["plan"].
    - Query the Pinecone index with semantic search and metadata filters.
    - Apply context compression to reduce token noise.
    - Apply re-ranking to prioritize the most relevant results.
    - Return updated state with retrieved_chunks populated.
    - Log actions to the scratchpad.

    Returns:
        Dict with "retrieved_chunks" key containing a list of dicts,
        each with: content, relevance_score, source, page_number.
    """
    
    current_task = state["plan"][0]

    # --- 2. Embed query ---
    query_embedding = generate_embeddings(current_task)

    # --- 3. Query Pinecone ---
    results = index_name.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True
    )

    matches = results.get("matches", [])

    # --- 4. Extract chunks ---
    chunks = []
    for match in matches:
        metadata = match.get("metadata", {})

        chunks.append({
            "content": metadata.get("text", ""),
            "score": match.get("score", 0.0),
            "source": metadata.get("source", "unknown"),
            "page_number": metadata.get("page_number", None)
        })

    # # --- 5. Context compression ---
    # for chunk in chunks:
    #     chunk["content"] = compress_text(chunk["content"])

    # --- 6. Re-ranking (Cross Encoder) ---
    if chunks:
        pairs = [(current_task, chunk["content"]) for chunk in chunks]
        rerank_scores = reranker.predict(pairs)

        for chunk, score in zip(chunks, rerank_scores):
            chunk["relevance_score"] = float(score)

        # Sort by reranked score
        chunks = sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)

    # --- 7. Keep top N ---
    top_chunks = chunks[:5]

    # --- 8. Logging ---
    state["scratchpad"].append(
        f"[Retriever] Task: {current_task} | Retrieved: {len(top_chunks)} chunks"
    )

    # --- 9. Return updated state ---
    return {
        "retrieved_chunks": top_chunks,
        "scratchpad": state["scratchpad"]
    }

