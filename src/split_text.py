from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load all text files
text_loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
)

# Load all PDF files
pdf_loader = DirectoryLoader(
    "data",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

# Combine documents
documents = text_loader.load() + pdf_loader.load()

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# Print first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n----- Chunk {i + 1} -----")
    print(chunk.page_content)
    print(chunk.metadata)