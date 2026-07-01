import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import time

# ========================================
# SETUP
# ========================================
print("Setting up RAG system for evaluation...")
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
collection = chroma_client.create_collection("eval_rag")
embeddings = embedding_model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(embeddings=embeddings, documents=chunks, ids=ids)

groq_client = Groq(api_key="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")
print("System ready!\n")

# ========================================
# STEP 1: TEST QUESTIONS
# 7 questions FROM the paper (should answer)
# 3 questions NOT in paper (should say I don't know)
# ========================================
test_questions = [
    # Questions the system SHOULD answer
    {
        "question": "What is the Transformer model?",
        "should_answer": True,
        "topic": "Transformer architecture"
    },
    {
        "question": "What is multi-head attention?",
        "should_answer": True,
        "topic": "Multi-head attention"
    },
    {
        "question": "What is the BLEU score of the Transformer?",
        "should_answer": True,
        "topic": "Model performance"
    },
    {
        "question": "What are encoder and decoder stacks?",
        "should_answer": True,
        "topic": "Model architecture"
    },
    {
        "question": "What is positional encoding?",
        "should_answer": True,
        "topic": "Positional encoding"
    },
    {
        "question": "What optimizer was used for training?",
        "should_answer": True,
        "topic": "Training details"
    },
    {
        "question": "What datasets were used for training?",
        "should_answer": True,
        "topic": "Training data"
    },
    # Questions system should say I DON'T KNOW
    {
        "question": "What is the price of Bitcoin today?",
        "should_answer": False,
        "topic": "Irrelevant - Bitcoin"
    },
    {
        "question": "Who won the cricket world cup?",
        "should_answer": False,
        "topic": "Irrelevant - Cricket"
    },
    {
        "question": "What is the best recipe for biryani?",
        "should_answer": False,
        "topic": "Irrelevant - Food"
    }
]

# ========================================
# STEP 2: RAG FUNCTION
# ========================================
def ask_rag(question):
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)

    # Check relevance
    relevance_prompt = f"""You are a relevance checker.
Answer with only YES or NO.
YES = context contains enough information to answer
NO = context does not contain relevant information

Question: {question}
Context: {context}

Answer (YES or NO only):"""

    relevance_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": relevance_prompt}]
    )
    is_relevant = "YES" in relevance_response.choices[0].message.content.upper()

    if is_relevant:
        answer_prompt = f"""Use ONLY the context below to answer.
If not in context say: I don't have enough information.

Context: {context}
Question: {question}
Answer:"""

        answer_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": answer_prompt}]
        )
        return answer_response.choices[0].message.content, True

    return "I don't have enough information.", False

# ========================================
# STEP 3: RUN EVALUATION
# ========================================
print("="*60)
print("STARTING EVALUATION - Testing 10 questions")
print("="*60)

results = []
correct = 0
total = len(test_questions)

for i, test in enumerate(test_questions):
    print(f"\nQuestion {i+1}/{total}: {test['question']}")
    print(f"Topic: {test['topic']}")
    print(f"Expected to answer: {test['should_answer']}")

    answer, answered = ask_rag(test['question'])

    # Check if system behaved correctly
    if test['should_answer'] == True and answered == True:
        status = "✅ CORRECT - Answered relevant question"
        correct += 1
    elif test['should_answer'] == False and answered == False:
        status = "✅ CORRECT - Correctly said I don't know"
        correct += 1
    elif test['should_answer'] == True and answered == False:
        status = "❌ WRONG - Should have answered but didn't"
    else:
        status = "❌ WRONG - Should have said I don't know"

    print(f"System answered: {answered}")
    print(f"Status: {status}")
    print(f"Answer preview: {answer[:100]}...")

    results.append({
        "question": test['question'],
        "correct": status.startswith("✅"),
        "status": status
    })

    # Small delay to avoid API rate limits
    time.sleep(1)

# ========================================
# STEP 4: FINAL SCORE
# ========================================
accuracy = (correct / total) * 100

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"\nTotal Questions: {total}")
print(f"Correct Responses: {correct}")
print(f"Wrong Responses: {total - correct}")
print(f"\n🎯 FINAL ACCURACY SCORE: {accuracy:.1f}%")
print("\nDetailed Results:")
for i, result in enumerate(results):
    print(f"Q{i+1}: {result['status']}")