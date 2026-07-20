from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Load documents
text_loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
)

pdf_loader = DirectoryLoader(
    "data",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

documents = text_loader.load() + pdf_loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Create embedding for the first chunk
vector = embeddings.embed_query(chunks[0].page_content)

print(f"Embedding Dimension: {len(vector)}")
print(vector[:10])  # Display the first 10 values
st.divider()

st.subheader("💬 Ask Questions")

question = st.text_input(
    "Ask something about your PDF"
)


if question:

    docs = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )


    prompt = f"""
You are an AI assistant.

Answer the question using the context below.

If the context does not contain the answer,
use your own knowledge.

Context:

{context}


Question:

{question}


Answer:
"""


    response = llm.invoke(prompt)


    st.write("### Answer")

    st.write(response.content)