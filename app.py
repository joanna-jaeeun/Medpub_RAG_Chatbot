"""
MedPub RAG Chatbot - Medical Literature Q&A System
CRAG + Multi-Query / HyDE / Step-Back / Original
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, TypedDict

from Bio import Entrez, Medline
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.load import dumps, loads
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# ============================================================
# 0. Page Config
# ============================================================
st.set_page_config(
    page_title="🏥 MedPub RAG Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🏥 MedPub RAG Chatbot")
st.caption("CRAG-powered Medical Literature Q&A")

# ============================================================
# 1. Environment
# ============================================================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    st.warning("⚠️ OPENAI_API_KEY not found in .env file.")

os.environ["OPENAI_API_KEY"] = openai_api_key

# ============================================================
# 2. Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 🔑 API")
    email = st.text_input(
        "PubMed Email",
        value="your.email@example.com",
        help="Required for PubMed API"
    )

    st.markdown("### 🔍 Search")
    query_method = st.selectbox(
        "Query Method",
        ["original", "multiquery", "hyde", "stepback"],
        help="original: direct search | multiquery: 5 versions | hyde: hypothetical doc | stepback: abstract first"
    )

    max_results = st.slider(
        "Max Papers",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("From", min_value=2000, max_value=2025, value=2020)
    with col2:
        end_year = st.number_input("To", min_value=2000, max_value=2025, value=2024)

    st.markdown("### 🤖 Model")
    model_name = st.selectbox(
        "LLM",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    )

    st.markdown("---")
    st.markdown("**Query Methods:**")
    st.markdown("- `original`: Direct search")
    st.markdown("- `multiquery`: 5 perspectives")
    st.markdown("- `hyde`: Hypothetical doc")
    st.markdown("- `stepback`: Abstract first")

# ============================================================
# 3. Session State
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# 4. PubMed Search
# ============================================================
def search_pubmed(query: str, email: str, max_results: int, start_year: int, end_year: int) -> List[Dict]:
    Entrez.email = email
    date_filter = f" AND {start_year}:{end_year}[dp]"
    full_query = query + date_filter

    try:
        handle = Entrez.esearch(db="pubmed", term=full_query, retmax=max_results, sort="relevance")
        search_results = Entrez.read(handle)
        handle.close()

        id_list = search_results["IdList"]
        if not id_list:
            return []

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        papers = []
        records = list(Medline.parse(handle))
        handle.close()

        for record in records:
            paper = {
                "pmid": record.get("PMID", ""),
                "title": record.get("TI", "No title"),
                "abstract": record.get("AB", "No abstract available"),
                "authors": ", ".join(record.get("AU", ["No author"])),
                "journal": record.get("TA", "No journal"),
                "year": record.get("DP", "").split()[0] if record.get("DP") else "",
                "doi": next((ref for ref in record.get("AID", []) if ref.endswith("[doi]")), "").replace("[doi]", ""),
                "types": record.get("PT", []),
            }

            if not paper["abstract"] or paper["abstract"].strip() in {"", "No abstract available"}:
                try:
                    ah = Entrez.efetch(db="pubmed", id=paper["pmid"], rettype="abstract", retmode="text")
                    abstract_txt = ah.read()
                    ah.close()
                    if isinstance(abstract_txt, bytes):
                        abstract_txt = abstract_txt.decode("utf-8", errors="ignore")
                    if abstract_txt and abstract_txt.strip():
                        paper["abstract"] = abstract_txt.strip()
                except Exception:
                    pass

            paper["full_text"] = f"""
                Title: {paper['title']}
                Authors: {paper['authors']}
                Journal: {paper['journal']}
                Year: {paper['year']}
                PMID: {paper['pmid']}
                DOI: {paper['doi']}
                Abstract: {paper['abstract']}
            """
            papers.append(paper)

        return papers

    except Exception as e:
        st.error(f"PubMed search error: {str(e)}")
        return []

# ============================================================
# 5. Indexing (Chunking + FAISS)
# ============================================================
def chunk_papers(papers):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for paper in papers:
        texts = splitter.split_text(paper["full_text"])
        for t in texts:
            chunks.append({
                "text": t,
                "pmid": paper["pmid"],
                "title": paper["title"],
                "authors": paper["authors"],
                "journal": paper["journal"],
                "year": paper["year"],
                "abstract": paper["abstract"]
            })
    return chunks


def build_faiss(papers):
    chunks = chunk_papers(papers)
    vectorstore = FAISS.from_texts(
        texts=[c["text"] for c in chunks],
        embedding=OpenAIEmbeddings(api_key=openai_api_key),
        metadatas=[{
            "pmid": c["pmid"],
            "title": c["title"],
            "authors": c["authors"],
            "journal": c["journal"],
            "year": c["year"],
            "abstract": c["abstract"]
        } for c in chunks]
    )
    return vectorstore

# ============================================================
# 6. Query Methods
# ============================================================
def get_unique_union(documents: list):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


def query_multiquery(question, retriever):
    template = """You are an AI language model assistant. Generate five different versions 
    of the given question to retrieve relevant documents from a vector database.
    Provide these alternative questions separated by newlines.
    Original question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    return retrieval_chain.invoke({"question": question})


def query_stepback(question, retriever):
    examples = [
        {"input": "Could the members of The Police perform lawful arrests?",
         "output": "What can the members of The Police do?"},
        {"input": "Jan Sindel's was born in what country?",
         "output": "What is Jan Sindel's personal history?"},
    ]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at world knowledge. Step back and paraphrase 
        a question to a more generic step-back question."""),
        few_shot_prompt,
        ("user", "{question}"),
    ])
    generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    step_back_question = generate_queries_step_back.invoke({"question": question})
    normal_docs = retriever.invoke(question)
    step_back_docs = retriever.invoke(step_back_question)
    return get_unique_union([normal_docs, step_back_docs])


def query_hyde(question, retriever):
    template = """Please write a scientific paper passage to answer the question.
    Question: {question}
    Passage:"""
    prompt = ChatPromptTemplate.from_template(template)
    generate_docs = (
        prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )
    hypothetical_doc = generate_docs.invoke({"question": question})
    return retriever.invoke(hypothetical_doc)


def run_query(question, method, retriever):
    if method == "original":
        return retriever.invoke(question)
    if method == "multiquery":
        return query_multiquery(question, retriever)
    if method == "hyde":
        return query_hyde(question, retriever)
    if method == "stepback":
        return query_stepback(question, retriever)
    raise ValueError(f"Unknown method: {method}")

# ============================================================
# 7. Generation
# ============================================================
def generate_answer(question, docs):
    context = "\n\n".join([d.page_content for d in docs])
    template = """
You are a medical AI assistant that provides accurate information based on published medical research.
Answer the question using only the provided papers. Always cite PMID, author(s), and year.
If insufficient information, clearly state that.

Context:
{context}

Question:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    seen = set()
    sources = []
    for d in docs:
        pmid = d.metadata.get("pmid", "")
        if pmid not in seen:
            seen.add(pmid)
            sources.append({
                "title": d.metadata.get("title", ""),
                "authors": d.metadata.get("authors", ""),
                "journal": d.metadata.get("journal", ""),
                "year": d.metadata.get("year", ""),
                "pmid": pmid,
                "abstract": d.metadata.get("abstract", "")
            })
    return answer, sources

# ============================================================
# 8. CRAG (LangGraph)
# ============================================================

# Retrieval Grader
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


def build_crag_app(retriever_obj, query_method_val, max_results_val, start_year_val, end_year_val, email_val):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of a retrieved document to a user question.
        Give a binary score 'yes' or 'no'."""),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    retrieval_grader = grade_prompt | structured_llm_grader

    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question re-writer that converts an input question to a better version
        optimized for searching PubMed medical literature."""),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # GraphState
    class GraphState(TypedDict):
        question: str
        generation: str
        web_search: str
        documents: List[str]
        query_method: str

    # Node functions
    def retrieve(state):
        question = state["question"]
        method = state["query_method"]
        documents = run_query(question, method, retriever_obj)
        return {"documents": documents, "question": question}

    def generate(state):
        question = state["question"]
        documents = state["documents"]
        answer, sources = generate_answer(question, documents)
        return {"documents": documents, "question": question, "generation": answer, "sources": sources}

    def grade_documents(state):
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            if score.binary_score == "yes":
                filtered_docs.append(d)
            else:
                web_search = "Yes"
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def transform_query(state):
        question = state["question"]
        documents = state["documents"]
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def research_again(state):
        question = state["question"]
        documents = state["documents"]
        new_papers = search_pubmed(
            query=question,
            email=email_val,
            max_results=max_results_val,
            start_year=start_year_val,
            end_year=end_year_val
        )
        if new_papers:
            new_vs = build_faiss(new_papers)
            new_retriever = new_vs.as_retriever(search_kwargs={"k": 5})
            new_docs = new_retriever.invoke(question)
            documents.extend(new_docs)
        return {"documents": documents, "question": question}

    def decide_to_generate(state):
        web_search = state["web_search"]
        if web_search == "Yes":
            return "transform_query"
        else:
            return "generate"

    # Build graph
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("research_again", research_again)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )
    workflow.add_edge("transform_query", "research_again")
    workflow.add_edge("research_again", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# ============================================================
# 9. Main App
# ============================================================
def main():
    tab1, tab2 = st.tabs(["💬 Chatbot", "📚 Search History"])

    with tab1:
        st.markdown("### 💬 Medical Literature Q&A (CRAG)")

        # Reset button
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()

        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    st.markdown("---")
                    st.markdown("**📚 References:**")
                    for idx, source in enumerate(message["sources"], 1):
                        with st.expander(f"{idx}. {source['title']} (PMID: {source['pmid']})"):
                            st.markdown(f"**Authors:** {source['authors']}")
                            st.markdown(f"**Journal:** {source['journal']} ({source['year']})")
                            st.markdown(f"**PMID:** {source['pmid']}")
                            if source.get("abstract"):
                                st.markdown("**Abstract:**")
                                st.text_area("", source["abstract"], height=150,
                                           disabled=True, key=f"hist_abs_{idx}_{message['content'][:10]}")
                            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/"
                            st.markdown(f"[🔗 View on PubMed]({pubmed_url})")

        # Chat input
        if prompt := st.chat_input("Ask a medical question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching PubMed & generating answer (CRAG)..."):
                    try:
                        # 1. 질문으로 바로 PubMed 검색
                        papers = search_pubmed(
                            query=prompt,
                            email=email,
                            max_results=max_results,
                            start_year=start_year,
                            end_year=end_year
                        )

                        if not papers:
                            st.error("No papers found. Try a different question.")
                            return

                        # 2. FAISS 구축
                        vs = build_faiss(papers)
                        retriever = vs.as_retriever(search_kwargs={"k": 5})

                        # 3. CRAG 앱 빌드
                        app = build_crag_app(
                            retriever,
                            query_method,
                            max_results,
                            start_year,
                            end_year,
                            email
                        )

                        # 4. 실행
                        inputs = {
                            "question": prompt,
                            "query_method": query_method,
                            "generation": "",
                            "web_search": "No",
                            "documents": []
                        }

                        final_value = None
                        status_placeholder = st.empty()

                        for output in app.stream(inputs):
                            for key, value in output.items():
                                status_placeholder.info(f"⚙️ Running node: `{key}`")
                                final_value = value

                        status_placeholder.empty()

                        if final_value and "generation" in final_value:
                            answer = final_value["generation"]
                            sources = final_value.get("sources", [])

                            st.markdown(answer)

                            if sources:
                                st.markdown("---")
                                st.markdown("**📚 References:**")
                                for idx, source in enumerate(sources, 1):
                                    with st.expander(f"{idx}. {source['title']} (PMID: {source['pmid']})"):
                                        st.markdown(f"**Authors:** {source['authors']}")
                                        st.markdown(f"**Journal:** {source['journal']} ({source['year']})")
                                        if source.get("abstract"):
                                            st.markdown("**Abstract:**")
                                            st.text_area("", source["abstract"], height=150,
                                                       disabled=True, key=f"abs_{idx}_{prompt[:10]}")
                                        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/"
                                        st.markdown(f"[🔗 View on PubMed]({pubmed_url})")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                        else:
                            st.error("Failed to generate answer.")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with tab2:
        st.markdown("### 📚 Search History")
        if st.session_state.messages:
            qa_pairs = []
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    qa_pairs.append({"Q": msg["content"]})
                elif msg["role"] == "assistant" and qa_pairs:
                    qa_pairs[-1]["A"] = msg["content"][:200] + "..."

            for idx, qa in enumerate(qa_pairs, 1):
                with st.expander(f"Q{idx}: {qa.get('Q', '')}"):
                    st.markdown(f"**Answer preview:** {qa.get('A', 'N/A')}")
        else:
            st.info("No search history yet.")


if __name__ == "__main__":
    main()
