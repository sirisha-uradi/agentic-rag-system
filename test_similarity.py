from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Our question and 3 possible answers
question = "What is machine learning?"

sentences = [
    "Machine learning is a subset of artificial intelligence",
    "Artificial intelligence helps computers think like humans",
    "Pizza is a popular Italian food with cheese and tomato"
]

# Convert everything to embeddings
question_embedding = model.encode([question])
sentence_embeddings = model.encode(sentences)

# Measure similarity between question and each sentence
similarities = cosine_similarity(question_embedding, sentence_embeddings)

print("Question:", question)
print("\n--- SIMILARITY SCORES ---")
for i, sentence in enumerate(sentences):
    score = similarities[0][i]
    print(f"\nSentence {i+1}: {sentence}")
    print(f"Similarity score: {score:.4f}")

print("\n--- CONCLUSION ---")
best_match = similarities[0].argmax()
print(f"Most relevant sentence: {sentences[best_match]}")