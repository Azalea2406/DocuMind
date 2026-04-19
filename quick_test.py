# quick_test.py — full pipeline test without a real PDF

from structure_parser  import parse_structure
from embedder          import chunk_sections, build_index, search
from entity_extractor  import extract_entities_from_sections
from analyzer          import (summarize_document, analyze_document_risk,
                                answer_question, build_insight_report)

print("=" * 60)
print("DOCUMENT INTELLIGENCE — FULL PIPELINE TEST")
print("=" * 60)

# ── Fake document pages ────────────────────────────────────────────
pages = [
    "SERVICE AGREEMENT\n1. Definitions\nThe term Vendor refers to Acme Corp. The Client is John Smith.",
    "2. Payment Terms\nInvoices are due within 30 days. Late fees of 1.5% apply monthly.",
    "3. Liability\nVendor liability is unlimited and covers all damages including indirect losses.",
    "4. Termination\nEither party may terminate with 30 days written notice under laws of Delaware.",
    "5. Confidentiality\nBoth parties agree to keep all information confidential for 2 years."
]

# ── Step 1: Structure ──────────────────────────────────────────────
print("\n[1/6] Parsing structure...")
structure = parse_structure(pages)
print(f"      Title    : {structure.title}")
print(f"      Sections : {len(structure.sections)}")
for s in structure.sections:
    print(f"        - {s.heading}")

# ── Step 2: Chunking + Embedding ───────────────────────────────────
print("\n[2/6] Chunking and embedding...")
chunks = chunk_sections(structure.sections)
store  = build_index(chunks)
print(f"      Chunks created : {len(chunks)}")

# ── Step 3: Search ─────────────────────────────────────────────────
print("\n[3/6] Testing semantic search...")
results = search(store, "what are the payment terms?")
print(f"      Query: 'what are the payment terms?'")
for r in results:
    print(f"      [{r.section_heading}] {r.text[:80]}...")

# ── Step 4: Entity extraction ──────────────────────────────────────
print("\n[4/6] Extracting entities...")
entities = extract_entities_from_sections(structure.sections)
print(f"      Parties   : {[e.text for e in entities.parties]}")
print(f"      Dates     : {[e.text for e in entities.dates]}")
print(f"      Money     : {[e.text for e in entities.money]}")
print(f"      Custom    : {[e.text for e in entities.custom]}")

# ── Step 5: Summaries ──────────────────────────────────────────────
print("\n[5/6] Generating summaries (LLM call)...")
summaries = summarize_document(structure.sections, entities)
print(f"      Overall summary:")
print(f"      {summaries['overall'].get('overall_summary', 'N/A')}")
print(f"      Document type : {summaries['overall'].get('document_type', 'N/A')}")

# ── Step 6: Risk analysis ──────────────────────────────────────────
print("\n[6/6] Analysing risk (LLM call)...")
risks = analyze_document_risk(structure.sections, entities)
print(f"      Risk score : {risks['score']} / 100")
print(f"      Risk label : {risks['label'].upper()}")
for r in risks["section_risks"]:
    print(f"        - {r['section']}: {r['risk_level']} ({r['risk_score']}/100)")

# ── Final report ───────────────────────────────────────────────────
report = build_insight_report(structure.title, summaries, risks, entities, store)

print("\n" + "=" * 60)
print("PIPELINE TEST COMPLETE")
print(f"  Title       : {report['title']}")
print(f"  Risk        : {report['risk']['label'].upper()} ({report['risk']['score']}/100)")
print(f"  Sections    : {len(report['sections'])}")
print(f"  Parties     : {report['entities']['parties']}")
print("=" * 60)

# ── Q&A test ───────────────────────────────────────────────────────
print("\nQ&A TEST")
questions = [
    "What are the payment terms?",
    "Who are the parties involved?",
    "What happens if the contract is terminated?"
]
for q in questions:
    result = answer_question(store, q)
    print(f"\n  Q: {q}")
    print(f"  A: {result.get('answer', 'N/A')[:120]}...")
    print(f"     Confidence: {result.get('confidence')} | Source: {result.get('source_sections')}")

print("\nAll tests passed.")