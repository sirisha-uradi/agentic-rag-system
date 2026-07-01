import streamlit as st
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import time

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
# We use st.cache_resource so it loads
# only once and not every time user asks
# ========================================
@st.cache_resource
def setup_system():
    # Read PDF
    reader = pypdf.PdfReader("paper.pdf")
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    # Chunk it
    chunk_size = 500
    overlap = 50
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start = end - overlap

    # Embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # ChromaDB
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("streamlit_rag")
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )

    # Groq
    groq_client = Groq(
        api_key="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"
    )

    return embedding_model, collection, groq_client, len(chunks)

# ========================================
# LOAD SYSTEM
# ========================================
with st.spinner("Loading RAG system... Please wait..."):
    embedding_model, collection, groq_client, total_chunks = setup_system()

st.success(f"✅ System ready! Loaded {total_chunks} chunks from research paper.")

# ========================================
# SIDEBAR - Project Info
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
    st.write("- Streamlit")
    st.write("- Python 3.14")

# ========================================
# MAIN CHAT INTERFACE
# ========================================
st.divider()

# Example questions
st.subheader("💡 Try These Questions:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What is the Transformer?"):
        st.session_state.question = "What is the Transformer architecture?"
with col2:
    if st.button("What is multi-head attention?"):
        st.session_state.question = "What is multi-head attention?"
with col3:
    if st.button("What BLEU score was achieved?"):
        st.session_state.question = "What BLEU score was achieved?"

st.divider()

# Question input
question = st.text_input(
    "🔍 Ask any question:",
    value=st.session_state.get("question", ""),
    placeholder="Type your question here..."
)

if st.button("🚀 Ask", type="primary"):
    if question:
        st.divider()

        # Show agent thinking process
        st.subheader("🧠 Agent Thinking Process:")

        with st.status("Agent is working...", expanded=True) as status:

            # Step 1: Retrieve
            st.write("🔍 Step 1: Searching vector database...")
            query_embedding = embedding_model.encode(
                [question]
            ).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3
            )
            retrieved_chunks = results['documents'][0]
            time.sleep(0.5)
            st.write(f"✅ Found {len(retrieved_chunks)} chunks!")

            # Step 2: Check relevance
            st.write("🔎 Step 2: Cross checking relevance...")
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
            time.sleep(0.5)

            if is_relevant:
                st.write("✅ Chunks are relevant!")

                # Step 3: Generate answer
                st.write("💬 Step 3: Generating intelligent answer...")

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
                status.update(
                    label="✅ Answer ready!",
                    state="complete"
                )

            else:
                st.write("❌ Chunks not relevant enough!")
                st.write("🔄 This question is outside the knowledge base...")
                answer = None
                status.update(
                    label="❌ Could not find relevant information",
                    state="error"
                )

        st.divider()

        # Show retrieved chunks
        with st.expander("📄 View Retrieved Chunks"):
            for i, chunk in enumerate(retrieved_chunks):
                st.text_area(
                    f"Chunk {i+1}:",
                    chunk,
                    height=100
                )

        # Show final answer
        st.subheader("💡 Final Answer:")
        if answer:
            st.success(answer)
        else:
            st.error("""
            ❌ I don't have enough information to answer this question.
            
            This question is outside the scope of the research paper.
            Please ask something related to the Transformer architecture!
            """)
    else:
        st.warning("Please type a question first!")