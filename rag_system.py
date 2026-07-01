import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ========================================
# STEP 1: READ AND CHUNK THE PDF
# ========================================
print("Step 1: Reading PDF...")
reader = pypdf.PdfReader("paper.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

# Split into chunks
chunk_size = 500
overlap = 50
chunks = []
start = 0
while start < len(full_text):
    end = start + chunk_size
    chunks.append(full_text[start:end])
    start = end - overlap

print(f"Total chunks created: {len(chunks)}")

# ========================================
# STEP 2: STORE IN VECTOR DATABASE
# ========================================
print("\nStep 2: Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Storing chunks in ChromaDB...")
client = chromadb.Client()
collection = client.create_collection("rag_collection")

embeddings = embedding_model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(
    embeddings=embeddings,
    documents=chunks,
    ids=ids
)
print(f"Successfully stored {len(chunks)} chunks!")

# ========================================
# STEP 3: SETUP LLM
# ========================================
print("\nStep 3: Setting up LLM...")
groq_client = Groq(api_key="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")
print("LLM ready!")

# ========================================
# STEP 4: THE RAG FUNCTION
# ========================================
def ask_question(question):
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print('='*50)

    # Step A: Convert question to embedding
    question_embedding = embedding_model.encode([question]).tolist()

    # Step B: Find most relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)

    print(f"\nRetrieved {len(retrieved_chunks)} relevant chunks...")

    # Step C: Build prompt template
    prompt = f"""You are a helpful AI assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

    # Step D: Send to LLM and get answer
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    print(f"\nAnswer: {answer}")
    return answer

# ========================================
# STEP 5: ASK QUESTIONS!
# ========================================
print("\n" + "="*50)
print("RAG SYSTEM IS READY!")
print("="*50)

# Ask 3 different questions
ask_question("What is the Transformer architecture?")
ask_question("What is multi-head attention?")
ask_question("What are the results of the Transformer model?")