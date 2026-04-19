import os
import json
from dotenv import load_dotenv
load_dotenv()

# ── LLM client — supports both OpenAI and Groq 
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"

if USE_GROQ:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    MODEL  = "llama-3.1-8b-instant"
else:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL  = "gpt-4o-mini"

from embedder        import VectorStore, search
from entity_extractor import EntityReport

# ── Shared helpers ─────────────────────────────────────────────────

def llm(system: str, user: str, temperature=0.2) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
    )
    return response.choices[0].message.content.strip()

def safe_json(raw: str) -> dict:
    """Extract JSON even if the LLM wraps it in markdown fences."""
    import re
    raw = raw.strip()
    # strip ```json ... ``` fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)

# ── Q&A Engine (RAG) ───────────────────────────────────────────────

def answer_question(store: VectorStore, question: str) -> dict:
    chunks  = search(store, question, top_k=5)
    context = "\n\n".join(
        f"[{c.section_heading}, page {c.page_hint}]\n{c.text}"
        for c in chunks
    )
    SYSTEM = """You are a document analyst. Answer questions using ONLY
the provided context. If the answer is not in the context, say so.
Always cite which section your answer comes from."""

    USER = f"""Context:
{context}

Question: {question}

Reply in JSON:
{{
  "answer": "...",
  "confidence": "high|medium|low",
  "source_sections": ["..."],
  "caveat": null
}}"""

    raw = llm(SYSTEM, USER)
    try:
        return safe_json(raw)
    except Exception:
        return {"answer": raw, "confidence": "low",
                "source_sections": [], "caveat": None}

# ── Summary Generator ──────────────────────────────────────────────

def summarize_section(heading: str, content: str) -> dict:
    if len(content.strip()) < 40:
        return {"heading": heading, "summary": content, "key_points": []}

    SYSTEM = "You are a legal and business document summarizer. Be concise and precise."
    USER   = f"""Summarize this section.

Section: {heading}
Content: {content[:3000]}

Reply in JSON:
{{
  "summary": "2-3 sentence summary",
  "key_points": ["point 1", "point 2", "point 3"]
}}"""

    raw = llm(SYSTEM, USER)
    try:
        result = safe_json(raw)
        return {"heading": heading, **result}
    except Exception:
        return {"heading": heading, "summary": raw, "key_points": []}


def summarize_document(sections, entities: EntityReport) -> dict:
    section_summaries = [
        summarize_section(s.heading, s.content)
        for s in sections if s.content.strip()
    ]
    combined = "\n".join(
        f"- {s['heading']}: {s['summary']}" for s in section_summaries
    )
    SYSTEM = "You are a senior document analyst producing executive summaries."
    USER   = f"""Section summaries:
{combined}

Key parties: {[e.text for e in entities.parties[:5]]}
Financial terms: {[e.text for e in entities.money[:5]]}
Key dates: {[e.text for e in entities.dates[:5]]}

Reply in JSON:
{{
  "overall_summary": "3-4 sentence executive summary",
  "document_type": "contract|report|agreement|policy|other",
  "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "obligations": ["obligation 1", "obligation 2"]
}}"""

    raw = llm(SYSTEM, USER)
    try:
        overall = safe_json(raw)
    except Exception:
        overall = {"overall_summary": raw, "document_type": "other",
                   "key_points": [], "obligations": []}
    return {"overall": overall, "sections": section_summaries}

# ── Risk Analyser ──────────────────────────────────────────────────

RISK_SYSTEM = """You are a contract risk analyst. Classify clauses and
flag issues. Be conservative — if unsure, flag it.

Risk levels:
  critical — uncapped liability, auto-renewal traps, unilateral termination,
             IP ownership ambiguity
  review   — unusual payment terms, vague deliverables, broad indemnity
  safe     — standard, balanced, or boilerplate clauses"""


def analyze_section_risk(heading: str, content: str) -> dict:
    USER = f"""Analyze this clause for risk.

Section: {heading}
Content: {content[:2000]}

Reply in JSON:
{{
  "risk_level": "safe",
  "risk_score": 10,
  "flags": [
    {{"issue": "short description", "reason": "why risky", "severity": "low|medium|high"}}
  ],
  "recommendation": "what to do"
}}"""

    raw = llm(RISK_SYSTEM, USER)
    try:
        result = safe_json(raw)
        return {"section": heading, **result}
    except Exception:
        return {"section": heading, "risk_level": "review",
                "risk_score": 50, "flags": [], "recommendation": raw}


def analyze_document_risk(sections, entities: EntityReport) -> dict:
    section_risks = [
        analyze_section_risk(s.heading, s.content)
        for s in sections if s.content.strip()
    ]
    scores = [r.get("risk_score", 0) for r in section_risks]
    levels = [r.get("risk_level", "safe") for r in section_risks]

    if "critical" in levels:
        doc_score = max(scores)
    elif scores:
        doc_score = int(sum(scores) / len(scores))
    else:
        doc_score = 0

    if doc_score >= 70 or "critical" in levels:
        doc_label = "critical"
    elif doc_score >= 40 or "review" in levels:
        doc_label = "review"
    else:
        doc_label = "safe"

    return {
        "score":  doc_score,
        "label":  doc_label,
        "critical_flags": [
            {"section": r["section"], "flags": r.get("flags", [])}
            for r in section_risks if r.get("risk_level") == "critical"
        ],
        "section_risks": section_risks
    }

# ── Final report builder ───────────────────────────────────────────

def build_insight_report(title, summaries, risks,
                         entities: EntityReport, store: VectorStore) -> dict:
    return {
        "title":   title,
        "summary": summaries["overall"],
        "risk": {
            "score":          risks["score"],
            "label":          risks["label"],
            "critical_flags": risks["critical_flags"]
        },
        "entities": {
            "parties":    [e.text for e in entities.parties[:8]],
            "dates":      [e.text for e in entities.dates[:8]],
            "money":      [e.text for e in entities.money[:8]],
            "legal_refs": [e.text for e in entities.legal_refs[:8]],
            "custom":     [e.text for e in entities.custom[:8]]
        },
        "sections": [
            {
                "heading":    sec["heading"],
                "summary":    sec["summary"],
                "key_points": sec.get("key_points", []),
                "risk_level": risk.get("risk_level", "safe"),
                "risk_score": risk.get("risk_score", 0),
                "flags":      risk.get("flags", [])
            }
            for sec, risk in zip(summaries["sections"], risks["section_risks"])
        ]
    }
