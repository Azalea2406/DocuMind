import streamlit as st
import tempfile, os, json
from dotenv import load_dotenv
load_dotenv()

from parser            import parse_pdf
from structure_parser  import parse_structure
from embedder          import chunk_sections, build_index
from entity_extractor  import extract_entities_from_sections
from analyzer          import (summarize_document, analyze_document_risk,
                                answer_question, build_insight_report)

st.set_page_config(page_title="Doc Intelligence", layout="wide")
st.title("Document Intelligence System")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    run_btn  = st.button("Analyze", type="primary", disabled=uploaded is None)

if "report"  not in st.session_state: st.session_state.report  = None
if "store"   not in st.session_state: st.session_state.store   = None
if "history" not in st.session_state: st.session_state.history = []

# ── Pipeline ───────────────────────────────────────────────────────
if run_btn and uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        tmp_path = f.name

    try:
        with st.spinner("Parsing PDF..."):
            doc       = parse_pdf(tmp_path)
            structure = parse_structure(doc.pages)

        with st.spinner("Embedding chunks..."):
            chunks = chunk_sections(structure.sections)
            store  = build_index(chunks)

        with st.spinner("Extracting entities..."):
            entities = extract_entities_from_sections(structure.sections)

        with st.spinner("Generating summaries..."):
            summaries = summarize_document(structure.sections, entities)

        with st.spinner("Analysing risk..."):
            risks = analyze_document_risk(structure.sections, entities)

        report = build_insight_report(
            structure.title, summaries, risks, entities, store
        )
        st.session_state.report  = report
        st.session_state.store   = store
        st.session_state.history = []

    except Exception as e:
        st.error(f"Error during analysis: {e}")
    finally:
        os.unlink(tmp_path)

# ── Results ────────────────────────────────────────────────────────
report = st.session_state.report
if report:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Sections", "Risk", "Entities", "Ask a Question"
    ])

    # Tab 1: Overview
    with tab1:
        st.subheader(report["title"])
        risk  = report["risk"]
        color = {"safe": "green", "review": "orange", "critical": "red"}.get(risk["label"], "gray")
        col1, col2 = st.columns(2)
        col1.metric("Risk score", f"{risk['score']} / 100")
        col2.markdown(f"**Risk label:** :{color}[{risk['label'].upper()}]")

        st.markdown("### Executive summary")
        st.write(report["summary"].get("overall_summary", ""))

        st.markdown("### Key points")
        for pt in report["summary"].get("key_points", []):
            st.markdown(f"- {pt}")

        if report["summary"].get("obligations"):
            st.markdown("### Obligations")
            for ob in report["summary"]["obligations"]:
                st.markdown(f"- {ob}")

        st.markdown("### Download report")
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name="insights.json",
            mime="application/json"
        )

    # Tab 2: Sections
    with tab2:
        for sec in report["sections"]:
            badge = {"safe": "🟢", "review": "🟡", "critical": "🔴"}.get(sec["risk_level"], "⚪")
            with st.expander(f"{badge}  {sec['heading']}"):
                st.write(sec["summary"])
                if sec.get("key_points"):
                    st.markdown("**Key points**")
                    for pt in sec["key_points"]:
                        st.markdown(f"- {pt}")
                if sec.get("flags"):
                    st.markdown("**Risk flags**")
                    for flag in sec["flags"]:
                        st.warning(f"**{flag['issue']}** — {flag['reason']}")

    # Tab 3: Risk
    with tab3:
        st.markdown("### Critical flags")
        if not report["risk"]["critical_flags"]:
            st.success("No critical flags found.")
        else:
            for item in report["risk"]["critical_flags"]:
                st.error(f"**{item['section']}**")
                for flag in item["flags"]:
                    st.markdown(f"- **{flag['issue']}**: {flag['reason']}")

        st.markdown("### Section risk breakdown")
        for sec in report["sections"]:
            st.markdown(f"**{sec['heading']}**")
            # progress expects float 0.0–1.0
            score_float = min(max(sec["risk_score"] / 100, 0.0), 1.0)
            st.progress(score_float,
                        text=f"{sec['risk_level']} ({sec['risk_score']}/100)")

    # Tab 4: Entities
    with tab4:
        ents = report["entities"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Parties**")
            for e in ents["parties"]:    st.markdown(f"- {e}")
            st.markdown("**Dates**")
            for e in ents["dates"]:      st.markdown(f"- {e}")
            st.markdown("**Legal refs**")
            for e in ents["legal_refs"]: st.markdown(f"- {e}")
        with col2:
            st.markdown("**Money / Amounts**")
            for e in ents["money"]:      st.markdown(f"- {e}")
            st.markdown("**Custom**")
            for e in ents["custom"]:     st.markdown(f"- {e}")

    # Tab 5: Q&A
    with tab5:
        st.markdown("Ask anything about the document.")
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("e.g. What are the payment terms?")
        if question:
            st.session_state.history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    result = answer_question(st.session_state.store, question)
                ans  = result.get("answer", "")
                conf = result.get("confidence", "")
                srcs = result.get("source_sections", [])
                st.write(ans)
                if srcs:
                    st.caption(f"Confidence: {conf} · Sources: {', '.join(srcs)}")
                if result.get("caveat"):
                    st.info(result["caveat"])
            st.session_state.history.append({"role": "assistant", "content": ans})

else:
    st.info("Upload a PDF in the sidebar to get started.")
