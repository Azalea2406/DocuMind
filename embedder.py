import os
import pickle
import numpy as np
import faiss
import tiktoken
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
EMBED_DIM    = 384
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
enc         = tiktoken.get_encoding("cl100k_base")

# ── Data classes (defined BEFORE they are used) ───────────────────

@dataclass
class Chunk:
    text: str
    section_heading: str
    page_hint: int
    chunk_index: int
    token_count: int

@dataclass
class VectorStore:
    index: object   # faiss.IndexFlatIP
    chunks: list

# ── Chunking ───────────────────────────────────────────────────────

def chunk_sections(sections, chunk_size=400, overlap=80) -> list:
    chunks = []
    idx = 0
    for section in sections:
        tokens = enc.encode(section.content)
        if not tokens:
            continue
        start = 0
        while start < len(tokens):
            end    = min(start + chunk_size, len(tokens))
            window = tokens[start:end]
            chunks.append(Chunk(
                text            = enc.decode(window),
                section_heading = section.heading,
                page_hint       = section.page_hint,
                chunk_index     = idx,
                token_count     = len(window)
            ))
            idx += 1
            if end == len(tokens):
                break
            start += chunk_size - overlap
    return chunks

# ── Embedding ──────────────────────────────────────────────────────
def embed_texts(texts: list) -> np.ndarray:
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        raise ValueError("No text to embed — all chunks were empty.")
    vectors = _embed_model.encode(texts, show_progress_bar=False)
    return np.array(vectors, dtype=np.float32)


# ── FAISS index ────────────────────────────────────────────────────

def build_index(chunks: list) -> VectorStore:
    if not chunks:
        raise ValueError("No chunks — document may be empty.")

    filtered_chunks = [c for c in chunks if c.text.strip()]
    texts           = [c.text for c in filtered_chunks]

    if not texts:
        raise ValueError("All chunks were empty after filtering.")

    vectors = embed_texts(texts)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBED_DIM) 
    index.add(vectors)
    return VectorStore(index=index, chunks=filtered_chunks)

def save_store(store: VectorStore, path: str):
    faiss.write_index(store.index, f"{path}.faiss")
    with open(f"{path}.chunks", "wb") as f:
        pickle.dump(store.chunks, f)

def load_store(path: str) -> VectorStore:
    index  = faiss.read_index(f"{path}.faiss")
    with open(f"{path}.chunks", "rb") as f:
        chunks = pickle.load(f)
    return VectorStore(index=index, chunks=chunks)

# ── Retrieval ──────────────────────────────────────────────────────

def search(store: VectorStore, query: str, top_k=5) -> list:
    vec = embed_texts([query])
    faiss.normalize_L2(vec)
    scores, indices = store.index.search(vec, top_k)
    return [store.chunks[i] for i in indices[0] if i != -1]
