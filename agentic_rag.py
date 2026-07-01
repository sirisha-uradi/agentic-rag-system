import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ========================================
# SETUP - Same as before
# ========================================
print("Setting up RAG system...")
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
collection = chroma_client.create_collection("agentic_rag")

embeddings = embedding_model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(embeddings=embeddings, documents=chunks, ids=ids)

groq_client = Groq(api_key="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")
print("System ready!\n")

# ========================================
# STEP 1: RELEVANCE CHECKER
# This is the "cross checker" you described!
# ========================================
def is_relevant(question, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    
    prompt = f"""You are a relevance checker.
Look at this question and the retrieved context below.
Answer with only YES or NO.
YES = the context contains enough information to answer the question
NO = the context does not contain relevant information

Question: {question}
Context: {context}

Answer (YES or NO only):"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content.strip()
    return "YES" in answer.upper()

# ========================================
# STEP 2: QUERY REPHRASER
# This is the "refresh" you described!
# ========================================
def rephrase_query(original_question, attempt):
    prompt = f"""Rephrase this question in a different way to search for information.
Make it attempt number {attempt} so make it increasingly different.
Return ONLY the rephrased question, nothing else.

Original question: {original_question}
Rephrased question:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ========================================
# STEP 3: THE AGENTIC LOOP
# This is the heart of the system!
# ========================================
def agentic_ask(question):
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print('='*50)
    
    current_query = question
    max_attempts = 3
    
    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt}: Searching with query:")
        print(f"'{current_query}'")
        
        # Search the vector database
        query_embedding = embedding_model.encode(
            [current_query]
        ).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        retrieved_chunks = results['documents'][0]
        
        # Cross check relevance
        print("Cross checking relevance...")
        relevant = is_relevant(question, retrieved_chunks)
        
        if relevant:
            print("✅ Relevant chunks found! Generating answer...")
            
            context = "\n\n".join(retrieved_chunks)
            prompt = f"""You are a helpful AI assistant.
Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response.choices[0].message.content
            print(f"\n✅ FINAL ANSWER: {answer}")
            return answer
            
        else:
            print(f"❌ Not relevant enough.")
            
            if attempt < max_attempts:
                print("Rephrasing query and trying again...")
                current_query = rephrase_query(question, attempt + 1)
            else:
                print("\n❌ Could not find relevant information after 3 attempts.")
                print("FINAL ANSWER: I don't have enough information in my knowledge base to answer this question confidently.")
                return "I don't know"

# ========================================
# STEP 4: TEST IT!
# ========================================

# This should FIND the answer (relevant question)
agentic_ask("How does attention mechanism work in Transformers?")

# This should say I don't know (irrelevant question)
agentic_ask("What is the price of Bitcoin today?")