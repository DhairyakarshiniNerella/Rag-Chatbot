from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing vector database
vector_db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings,
)

# User question
query = "What is RAG?"

# Search similar chunks
results = vector_db.similarity_search(query, k=3)

print(f"Question: {query}\n")

for i, doc in enumerate(results, start=1):
    print(f"========== Result {i} ==========")
    print(doc.page_content)
    print("\nMetadata:", doc.metadata)
    print("-" * 50)