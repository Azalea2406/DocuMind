import re
import spacy
from dataclasses import dataclass, field
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

# ── Data models ────────────────────────────────────────────────────

@dataclass
class Entity:
    text: str
    label: str
    count: int
    first_seen_section: str
    context_snippets: list[str] = field(default_factory=list)

@dataclass
class EntityReport:
    parties: list[Entity]       # ORG + PERSON
    dates: list[Entity]         # DATE
    money: list[Entity]         # MONEY + PERCENT
    locations: list[Entity]     # GPE
    legal_refs: list[Entity]    # LAW
    custom: list[Entity]        # domain-specific (see below)


    # ── Custom pattern matcher ─────────────────────────────────────────
# spaCy misses domain-specific patterns like clause refs, case numbers,
# governing law clauses. Add them here with regex.

CUSTOM_PATTERNS = {
    "CLAUSE_REF":   re.compile(r'\b(clause|section|article|exhibit|schedule)\s+[\d\.]+[a-z]?\b', re.I),
    "CASE_NUMBER":  re.compile(r'\bCase\s+No\.?\s*[\w\-]+', re.I),
    "GOVERNING_LAW":re.compile(r'(laws?\s+of\s+(?:the\s+)?(?:State\s+of\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.I),
    "DURATION":     re.compile(r'\b\d+[\-\s]?(day|month|year|week)s?\b', re.I),
    "EMAIL":        re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'),
}

def extract_custom(text: str, section_heading: str) -> list[Entity]:
    found = []
    for label, pattern in CUSTOM_PATTERNS.items():
        matches = pattern.findall(text)
        counts = defaultdict(int)
        for m in matches:
            m_text = m if isinstance(m, str) else m[0]
            counts[m_text.strip()] += 1
        for m_text, count in counts.items():
            found.append(Entity(
                text=m_text,
                label=label,
                count=count,
                first_seen_section=section_heading
            ))
    return found


# ── spaCy extraction with context ─────────────────────────────────

LABEL_GROUPS = {
    "parties":    {"PERSON", "ORG"},
    "dates":      {"DATE"},
    "money":      {"MONEY", "PERCENT"},
    "locations":  {"GPE"},
    "legal_refs": {"LAW"},
}

def extract_entities_from_sections(sections) -> EntityReport:
    bucket: dict[str, dict[str, Entity]] = {k: {} for k in LABEL_GROUPS}
    custom_all: dict[str, Entity] = {}

    for section in sections:
        text = section.content
        if not text.strip():
            continue

        # spaCy pass
        doc = nlp(text[:50000])
        for ent in doc.ents:
            for group, labels in LABEL_GROUPS.items():
                if ent.label_ in labels:
                    key = ent.text.strip().lower()
                    if key not in bucket[group]:
                        # grab ~60 chars of surrounding context
                        start = max(0, ent.start_char - 40)
                        end   = min(len(text), ent.end_char + 40)
                        snippet = "…" + text[start:end].strip() + "…"
                        bucket[group][key] = Entity(
                            text=ent.text.strip(),
                            label=ent.label_,
                            count=1,
                            first_seen_section=section.heading,
                            context_snippets=[snippet]
                        )
                    else:
                        bucket[group][key].count += 1

        # Custom pattern pass
        for ent in extract_custom(text, section.heading):
            key = ent.text.lower()
            if key not in custom_all:
                custom_all[key] = ent
            else:
                custom_all[key].count += ent.count

    def sorted_by_count(d):
        return sorted(d.values(), key=lambda e: -e.count)

    return EntityReport(
        parties   =sorted_by_count(bucket["parties"]),
        dates     =sorted_by_count(bucket["dates"]),
        money     =sorted_by_count(bucket["money"]),
        locations =sorted_by_count(bucket["locations"]),
        legal_refs=sorted_by_count(bucket["legal_refs"]),
        custom    =sorted_by_count(custom_all),
    )


# ── Relationship inference ─────────────────────────────────────────
# A lightweight pass that spots "X shall pay Y $Z" style patterns —
# useful for the risk layer later.

RELATION_PATTERNS = [
    re.compile(r'([\w\s]+?)\s+shall\s+(pay|deliver|provide|indemnify)\s+([\w\s,\$]+)', re.I),
    re.compile(r'([\w\s]+?)\s+is\s+responsible\s+for\s+([\w\s,]+)', re.I),
    re.compile(r'([\w\s]+?)\s+agrees?\s+to\s+([\w\s,]+)', re.I),
]

@dataclass
class Relation:
    subject: str
    predicate: str
    obj: str
    source_section: str

def extract_relations(sections) -> list[Relation]:
    relations = []
    for section in sections:
        for pattern in RELATION_PATTERNS:
            for match in pattern.finditer(section.content):
                groups = match.groups()
                relations.append(Relation(
                    subject=groups[0].strip(),
                    predicate=groups[1].strip() if len(groups) > 1 else "",
                    obj=groups[2].strip() if len(groups) > 2 else "",
                    source_section=section.heading
                ))
    return relations[:30]   # cap to avoid noise