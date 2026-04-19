""" Extraction Module"""
import pdfplumber
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParsedDocument:
    raw_text: str
    pages: list[str]
    metadata: dict

def extract_with_pdfplumber(filepath: str) -> Optional[ParsedDocument]:
    pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
        if not pages:
            return None
        return ParsedDocument(
            raw_text="\n\n".join(pages),
            pages=pages,
            metadata={"page_count": len(pages), "parser": "pdfplumber"}
        )
    except Exception:
        return None

def extract_with_pymupdf(filepath: str) -> ParsedDocument:
    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        pages.append(page.get_text("text").strip())
    doc.close()
    return ParsedDocument(
        raw_text="\n\n".join(pages),
        pages=pages,
        metadata={"page_count": len(pages), "parser": "pymupdf"}
    )

def parse_pdf(filepath: str) -> ParsedDocument:
    result = extract_with_pdfplumber(filepath)
    if result and result.raw_text.strip():
        return result
    return extract_with_pymupdf(filepath)  # fallback