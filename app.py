import streamlit as st
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import time
from tools import web_search   # <-- NEW: our real tool

# ========================================
# PAGE SETUP
# ========================================
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG System")
st.subheader("Ask questions from any research paper intelligently!")

# ========================================
# LOAD EVERYTHING ONCE
# ========================================
@st.cache_resource
def setup_system():
    reader = pypdf.PdfReader("paper.pdf")
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    chunk_size = 500
    overlap = 50
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start = end - overlap

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("streamlit_rag")
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(embeddings=embeddings, documents=chunks, ids=ids)

    groq_client = Groq(api_key="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")

    return embedding_model, collection, groq_client, len(chunks)

with st.spinner("Loading RAG system... Please wait..."):
    embedding_model, collection, groq_client, total_chunks = setup_system()

st.success(f"✅ System ready! Loaded {total_chunks} chunks from research paper.")

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.header("📊 Project Info")
    st.info("""
    **Agentic RAG System**

    Built by: Sirisha Uradi! 🎓

    Paper: Attention Is All You Need

    Features:
    - ✅ PDF Processing
    - ✅ Vector Search
    - ✅ Agentic Loop
    - ✅ Self Correction
    - ✅ Web Search Tool (NEW)
    - ✅ Evaluation Harness
    """)

    st.header("📈 Evaluation Score")
    st.metric("Accuracy", "100%", "10/10 correct")
    st.metric("Relevant Questions", "7/7", "All answered")
    st.metric("Irrelevant Questions", "3/3", "All rejected")

    st.header("🔧 Tech Stack")
    st.write("- Groq (Llama 3.1)")
    st.write("- ChromaDB")
    st.write("- Sentence Transformers")
    st.write("- DuckDuckGo Search (tool)")
    st.write("- Streamlit")
    st.write("- Python 3.14")

# ========================================
# MAIN CHAT INTERFACE
# ========================================
st.divider()

st.subheader("💡 Try These Questions:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What is the Transformer?"):
        st.session_state.question = "What is the Transformer architecture?"
with col2:
    if st.button("What is multi-head attention?"):
        st.session_state.question = "What is multi-head attention?"
with col3:
    if st.button("Who won the 2026 FIFA World Cup?"):
        st.session_state.question = "Who won the 2026 FIFA World Cup?"

st.divider()

question = st.text_input(
    "🔍 Ask any question:",
    value=st.session_state.get("question", ""),
    placeholder="Type your question here..."
)

if st.button("🚀 Ask", type="primary"):
    if question:
        st.divider()
        st.subheader("🧠 Agent Thinking Process:")

        answer = None
        source_used = None

        with st.status("Agent is working...", expanded=True) as status:

            # ---- STEP 1: Search the paper ----
            st.write("🔍 Step 1: Searching the research paper...")
            query_embedding = embedding_model.encode([question]).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=3)
            retrieved_chunks = results['documents'][0]
            time.sleep(0.4)
            st.write(f"✅ Found {len(retrieved_chunks)} chunks from the paper.")

            # ---- STEP 2: Check relevance ----
            st.write("🔎 Step 2: Checking if the paper actually answers this...")
            context = "\n\n".join(retrieved_chunks)

            relevance_prompt = f"""Answer with only YES or NO.
YES = context contains enough information to answer
NO = context does not contain relevant information

Question: {question}
Context: {context}
Answer:"""

            relevance_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": relevance_prompt}]
            )
            is_relevant = "YES" in relevance_response.choices[0].message.content.upper()
            time.sleep(0.4)

            if is_relevant:
                # ---- PATH A: paper has the answer ----
                st.write("✅ Paper is relevant! No tool call needed.")
                st.write("💬 Step 3: Generating answer from the paper...")

                answer_prompt = f"""You are a helpful AI assistant.
Use ONLY the context below to answer the question.
Give a clear and detailed answer.

Context:
{context}

Question: {question}
Answer:"""

                answer_response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": answer_prompt}]
                )
                answer = answer_response.choices[0].message.content
                source_used = "📄 Research Paper"
                status.update(label="✅ Answer ready! (from paper)", state="complete")

            else:
                # ---- PATH B: paper doesn't know -> AGENT DECIDES to call a tool ----
                st.write("❌ Paper doesn't cover this.")
                st.write("🤖 **Agent decision:** calling `web_search` tool to find a real answer...")

                web_results = web_search(question, max_results=3)
                time.sleep(0.4)
                st.write("🌐 Tool returned live web results!")

                st.write("💬 Step 3: Generating answer from web search results...")

                web_answer_prompt = f"""You are a helpful AI assistant.
Use the web search results below to answer the question clearly and concisely.
Mention that this information came from a live web search, not the research paper.

Web Search Results:
{web_results}

Question: {question}
Answer:"""

                web_answer_response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": web_answer_prompt}]
                )
                answer = web_answer_response.choices[0].message.content
                source_used = "🌐 Live Web Search (tool call)"
                status.update(label="✅ Answer ready! (via web_search tool)", state="complete")

        st.divider()

        with st.expander("📄 View Retrieved Chunks (from paper)"):
            for i, chunk in enumerate(retrieved_chunks):
                st.text_area(f"Chunk {i+1}:", chunk, height=100)

        st.subheader("💡 Final Answer:")
        st.caption(f"Source used: {source_used}")
        if answer:
            st.success(answer)
        else:
            st.error("Something went wrong generating the answer.")
    else:
        st.warning("Please type a question first!")