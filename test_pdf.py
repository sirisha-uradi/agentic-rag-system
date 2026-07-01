import pypdf

# Step 1: Read the PDF
print("Reading PDF...")
reader = pypdf.PdfReader("paper.pdf")

# How many pages?
total_pages = len(reader.pages)
print(f"Total pages in PDF: {total_pages}")

# Step 2: Extract all text from all pages
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

print(f"Total characters extracted: {len(full_text)}")

# Step 3: Split into chunks
# We split every 500 characters with 50 character overlap
# Overlap means chunks share some text so nothing gets cut off awkwardly

chunk_size = 500
overlap = 50
chunks = []

start = 0
while start < len(full_text):
    end = start + chunk_size
    chunk = full_text[start:end]
    chunks.append(chunk)
    start = end - overlap

print(f"\nTotal chunks created: {len(chunks)}")
print("\n--- FIRST CHUNK ---")
print(chunks[0])
print("\n--- SECOND CHUNK ---")
print(chunks[1])