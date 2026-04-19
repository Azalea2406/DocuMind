import re
import spacy
from dataclasses import dataclass, field

nlp = spacy.load("en_core_web_sm")

@dataclass
class Section:
    heading: str
    level: int          # 1 = top-level, 2 = sub-heading
    content: str
    page_hint: int = 0  # which page this likely came from

@dataclass
class Entity:
    text: str
    label: str          # PERSON, ORG, DATE, MONEY, GPE, etc.
    count: int = 1

@dataclass
class DocumentStructure:
    title: str
    sections: list[Section]
    entities: list[Entity]
    raw_pages: list[str]

    # ── Heading detection ──────────────────────────────────────────────

HEADING_PATTERNS = [
    (1, re.compile(r'^([A-Z][A-Z\s]{3,50})$')),               # ALL CAPS LINE
    (1, re.compile(r'^(\d+\.\s+[A-Z].{3,60})$')),             # 1. Introduction
    (2, re.compile(r'^(\d+\.\d+\s+[A-Z].{3,50})$')),          # 1.1 Background
    (1, re.compile(r'^(Article\s+\d+[:\.]?.*)$', re.I)),       # Article 1: ...
    (1, re.compile(r'^(Section\s+\d+[:\.]?.*)$', re.I)),       # Section 3: ...
    (2, re.compile(r'^([A-Z][a-z].{3,50}):$')),                # Short Title:
]

def detect_heading(line: str) -> tuple[int, str] | None:
    line = line.strip()
    if len(line) < 3 or len(line) > 100:
        return None
    for level, pattern in HEADING_PATTERNS:
        if pattern.match(line):
            return level, line
    return None


def split_into_sections(pages: list[str]) -> list[Section]:
    sections = []
    current_heading = "Introduction"
    current_level = 1
    current_content_lines = []
    current_page = 0

    for page_num, page_text in enumerate(pages):
        for line in page_text.splitlines():
            result = detect_heading(line)
            if result:
                # Save previous section
                if current_content_lines:
                    sections.append(Section(
                        heading=current_heading,
                        level=current_level,
                        content=" ".join(current_content_lines).strip(),
                        page_hint=current_page
                    ))
                current_heading = result[1]
                current_level = result[0]
                current_content_lines = []
                current_page = page_num
            else:
                stripped = line.strip()
                if stripped:
                    current_content_lines.append(stripped)

    # Flush final section
    if current_content_lines:
        sections.append(Section(
            heading=current_heading,
            level=current_level,
            content=" ".join(current_content_lines).strip(),
            page_hint=current_page
        ))

    return sections

    # ── Entity extraction ──────────────────────────────────────────────

USEFUL_ENTITY_LABELS = {"PERSON", "ORG", "DATE", "MONEY", "GPE", "LAW", "PERCENT"}

def extract_entities(text: str) -> list[Entity]:
    doc = nlp(text[:50000])  # spaCy limit guard
    counts: dict[tuple, int] = {}
    for ent in doc.ents:
        if ent.label_ in USEFUL_ENTITY_LABELS:
            key = (ent.text.strip(), ent.label_)
            counts[key] = counts.get(key, 0) + 1

    return [
        Entity(text=k[0], label=k[1], count=v)
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    ]

# ── Main entry point ───────────────────────────────────────────────

def parse_structure(pages: list[str]) -> DocumentStructure:
    full_text = "\n".join(pages)

    # Try to pull a title from the first non-empty lines
    title = "Untitled Document"
    for line in pages[0].splitlines():
        line = line.strip()
        if len(line) > 5:
            title = line
            break

    sections = split_into_sections(pages)
    entities = extract_entities(full_text)

    return DocumentStructure(
        title=title,
        sections=sections,
        entities=entities,
        raw_pages=pages
    )