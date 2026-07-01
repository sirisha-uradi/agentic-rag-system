import pypdf
import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Read PDF
print("Reading PDF...")
reader = pypdf.PdfReader("paper.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

# Step 2: Split into chunks
chunk_size = 500
overlap = 50
chunks = []
start = 0
while start < len(full_text):
    end = start + chunk_size
    chunks.append(full_text[start:end])
    start = end - overlap

print(f"Total chunks: {len(chunks)}")

# Step 3: Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Create ChromaDB database
print("Creating vector database...")
client = chromadb.Client()
collection = client.create_collection("research_paper")

# Step 5: Store all chunks with embeddings
print("Storing chunks in database...")
embeddings = model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(
    embeddings=embeddings,
    documents=chunks,
    ids=ids
)

print(f"Successfully stored {len(chunks)} chunks!")

# Step 6: Ask a question!
print("\n--- ASKING A QUESTION ---")
question = "What is the Transformer architecture?"
question_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings=question_embedding,
    n_results=2
)

print(f"Question: {question}")
print(f"\nMost relevant chunk found:")
print(results['documents'][0][0])