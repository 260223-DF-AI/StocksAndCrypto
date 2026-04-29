"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
"""

import argparse
import os
from typing import List
# from langchain.schema import Document
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    """Parse ingestion CLI arguments."""
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing PDF/text documents.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="primary-corpus",
        help="Pinecone namespace to upsert into.",
    )
    return parser.parse_args()


def load_documents(input_dir: str) -> list:
    """
    Load and return raw documents from the input directory.

    TODO:
    - Support PDF files (e.g., using pypdf or LangChain's PyPDFLoader).
    - Support plain text files.
    - Return a list of Document objects with content and metadata
      (source filename, page number).
    """
    documents = []

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # ---- PDF handling ----
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()

            # Each page becomes a Document already
            for doc in pdf_docs:
                doc.metadata["source"] = filename
            documents.extend(pdf_docs)

        # ---- Text file handling ----
        elif filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": 0  # text files don’t have pages
                    }
                )
            )

    return documents

def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    TODO:
    - Use RecursiveCharacterTextSplitter or sentence-level splitting.
    - Attach chunk metadata (chunk_id, source, page_number, timestamp).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunked_docs = []
    chunk_id = 0

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": chunk_id,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", None),
                    "chunk_index": i
                }
            )
            chunked_docs.append(new_doc)
            chunk_id += 1

    return chunked_docs

def generate_embeddings(chunks: list) -> list:
    """
    Generate vector embeddings for document chunks in batches.

    TODO:
    - Use Sentence Transformers (e.g., all-MiniLM-L6-v2)
      or Bedrock Titan Embeddings.
    - Process in batches for efficiency (see W5 Monday — batch embedding).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in chunks]

    # Batch encode
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    results = []

    for i, (doc, embedding) in enumerate(zip(chunks, embeddings)):
        results.append({
            "id": str(doc.metadata.get("chunk_id", i)),
            "values": embedding.tolist(),
            "metadata": {
                "text": doc.page_content,
                **doc.metadata
            }
        })

    return results


def upsert_to_pinecone(embeddings: list, namespace: str) -> None:
    """
    Upsert embedding vectors and metadata into the Pinecone index.

    TODO:
    - Initialize the Pinecone client using env vars.
    - Upsert vectors with rich metadata into the specified namespace.
    """
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        raise ValueError("Missing Pinecone environment variables.")

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    if not pc.has_index(index_name):
        pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )


    index = pc.Index(index_name)

    # Batch upsert
    batch_size = 100

    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]

        index.upsert(
            vectors=batch,
            namespace=namespace
        )


def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()
    args = parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents)
    embeddings = generate_embeddings(chunks)
    #print(embeddings)
    upsert_to_pinecone(embeddings, args.namespace)

    print(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
