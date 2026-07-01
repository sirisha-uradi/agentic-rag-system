from sentence_transformers import SentenceTransformer

# Load the embedding model
# This will download automatically first time (about 90MB)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Let's create embeddings for 3 sentences
sentences = [
    "Artificial intelligence helps computers think like humans",
    "Machine learning is a subset of artificial intelligence", 
    "Pizza is a popular Italian food with cheese and tomato"
]

# Convert sentences to embeddings (secret code numbers)
embeddings = model.encode(sentences)

# Let's see what they look like
print("\n--- RESULTS ---")
print(f"Number of sentences: {len(sentences)}")
print(f"Each embedding has {len(embeddings[0])} numbers")
print(f"\nFirst 5 numbers of sentence 1: {embeddings[0][:5]}")
print(f"First 5 numbers of sentence 2: {embeddings[1][:5]}")
print(f"First 5 numbers of sentence 3: {embeddings[2][:5]}")