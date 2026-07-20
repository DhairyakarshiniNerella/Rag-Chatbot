from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)

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

# Load documents
text_documents = text_loader.load()
pdf_documents = pdf_loader.load()

# Combine all documents
documents = text_documents + pdf_documents

print(f"Total Documents Loaded: {len(documents)}")

for doc in documents:
    print(doc.metadata)