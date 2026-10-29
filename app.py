"""MedPub RAG Chatbot - Streamlit front end.

All retrieval and generation happens in the FastAPI service (api/main.py);
this file only collects the question and settings, calls POST /ask and
renders the answer with its PubMed sources.

    API_URL=http://localhost:8000 streamlit run app.py
"""
import datetime as dt
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
THIS_YEAR = dt.date.today().year

st.set_page_config(page_title="🏥 MedPub RAG Chatbot", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")
st.title("🏥 MedPub RAG Chatbot")
st.caption("CRAG-powered Medical Literature Q&A · answers cite PubMed papers · not medical advice")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    access_code = st.text_input("Access code", type="password",
                                help="Ask the owner for a code if the demo requires one")

    st.markdown("### 🔍 Search")
    query_method = st.selectbox(
        "Query Method", ["original", "multiquery", "hyde", "stepback"],
        help="original: direct search | multiquery: 5 versions | hyde: hypothetical doc | stepback: abstract first")
    max_results = st.slider("Max Papers", min_value=5, max_value=50, value=10, step=5)
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("From", min_value=1950, max_value=THIS_YEAR, value=2020)
    with col2:
        end_year = st.number_input("To", min_value=1950, max_value=THIS_YEAR, value=THIS_YEAR)

    st.markdown("### 🤖 Model")
    model_name = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o"])

    st.markdown("---")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API online · v{health.get('version', '?')}")
    except Exception:
        st.error(f"API not reachable at {API_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources, key_prefix):
    if not sources:
        return
    st.markdown("---")
    st.markdown("**📚 References:**")
    for idx, s in enumerate(sources, 1):
        with st.expander(f"{idx}. {s['title']} (PMID: {s['pmid']})"):
            st.markdown(f"**Authors:** {s['authors']}")
            st.markdown(f"**Journal:** {s['journal']} ({s['year']})")
            if s.get("abstract"):
                st.text_area("Abstract", s["abstract"], height=150, disabled=True,
                             key=f"{key_prefix}_{idx}")
            st.markdown(f"[🔗 View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/)")


def ask_api(question):
    payload = {"question": question, "query_method": query_method, "max_results": int(max_results),
               "start_year": int(start_year), "end_year": int(end_year), "model": model_name}
    headers = {"X-Access-Code": access_code} if access_code else {}
    resp = requests.post(f"{API_URL}/ask", json=payload, headers=headers, timeout=180)
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith(
            "application/json") else resp.text
        raise RuntimeError(f"{resp.status_code}: {detail}")
    return resp.json()


tab1, tab2 = st.tabs(["💬 Chatbot", "📚 Search History"])

with tab1:
    st.markdown("### 💬 Medical Literature Q&A (CRAG)")
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            show_sources(msg.get("sources"), key_prefix=f"hist{i}")

    if prompt := st.chat_input("Ask a medical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching PubMed & generating answer (CRAG)..."):
                try:
                    result = ask_api(prompt)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    result = None
            if result:
                st.markdown(result["answer"])
                st.caption(f"Nodes: {' → '.join(result['nodes'])} · {result['n_papers']} papers · "
                           f"{result['latency_s']}s")
                show_sources(result["sources"], key_prefix=f"new{len(st.session_state.messages)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]})

with tab2:
    st.markdown("### 📚 Search History")
    pairs = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            pairs.append({"Q": msg["content"]})
        elif pairs:
            pairs[-1]["A"] = msg["content"][:200] + "..."
    if not pairs:
        st.info("No search history yet.")
    for idx, qa in enumerate(pairs, 1):
        with st.expander(f"Q{idx}: {qa.get('Q', '')}"):
            st.markdown(f"**Answer preview:** {qa.get('A', 'N/A')}")
