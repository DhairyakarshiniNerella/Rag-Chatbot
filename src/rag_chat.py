from dotenv import load_dotenv
import os

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Load vector database
vector_db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings,
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# User question
query = "What is RAG?"

# Retrieve relevant documents
docs = vector_db.similarity_search(query, k=3)

# Combine retrieved context
context = "\n\n".join(doc.page_content for doc in docs)

# Prompt
prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

# Generate answer
response = llm.invoke(prompt)

print("Question:")
print(query)

print("\nAnswer:")
print(response.content)