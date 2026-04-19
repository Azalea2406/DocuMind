import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from parser           import parse_pdf
from structure_parser import parse_structure
from embedder         import chunk_sections, build_index, save_store, load_store, search
from entity_extractor import extract_entities_from_sections, extract_relations

app       = FastAPI()
UPLOAD_DIR = "uploads"
STORE_DIR  = "stores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STORE_DIR,  exist_ok=True)

def serialize(e):
    return {
        "text":       e.text,
        "type":       e.label,
        "mentions":   e.count,
        "first_seen": e.first_seen_section,
        "context":    e.context_snippets[:1]
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    file_id  = str(uuid.uuid4())
    filepath = f"{UPLOAD_DIR}/{file_id}.pdf"

    with open(filepath, "wb") as f:
        f.write(await file.read())

    doc       = parse_pdf(filepath)
    structure = parse_structure(doc.pages)
    chunks    = chunk_sections(structure.sections)
    store     = build_index(chunks)
    save_store(store, f"{STORE_DIR}/{file_id}")
    entities  = extract_entities_from_sections(structure.sections)
    relations = extract_relations(structure.sections)

    return JSONResponse({
        "file_id":       file_id,
        "title":         structure.title,
        "page_count":    doc.metadata["page_count"],
        "chunks_created":len(chunks),
        "sections": [
            {"heading": s.heading, "level": s.level,
             "preview": s.content[:200], "page": s.page_hint}
            for s in structure.sections
        ],
        "entities": {
            "parties":    [serialize(e) for e in entities.parties[:10]],
            "dates":      [serialize(e) for e in entities.dates[:10]],
            "money":      [serialize(e) for e in entities.money[:10]],
            "locations":  [serialize(e) for e in entities.locations],
            "legal_refs": [serialize(e) for e in entities.legal_refs[:10]],
            "custom":     [serialize(e) for e in entities.custom[:15]],
        },
        "relations": [
            {"subject": r.subject, "predicate": r.predicate,
             "object": r.obj, "section": r.source_section}
            for r in relations
        ]
    })

@app.get("/search/{file_id}")
async def search_doc(file_id: str, q: str, top_k: int = 5):
    store   = load_store(f"{STORE_DIR}/{file_id}")
    results = search(store, q, top_k)
    return {
        "query": q,
        "results": [
            {"section": c.section_heading, "page": c.page_hint,
             "text": c.text, "tokens": c.token_count}
            for c in results
        ]
    }
    
