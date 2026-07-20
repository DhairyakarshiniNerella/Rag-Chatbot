from dotenv import load_dotenv
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ------------------------
# Load Environment
# ------------------------
load_dotenv()


# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)
st.caption(
    "Chat with your uploaded documents using Groq + RAG"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions.")


# ------------------------
# Upload PDF
# ------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )


    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    st.success("PDF uploaded successfully!")


    # ------------------------
    # Load PDF
    # ------------------------
    loader = PyPDFLoader(file_path)

    documents = loader.load()


    st.write(
        f"Total Pages: {len(documents)}"
    )


    # ------------------------
    # Chunking
    # ------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )


    chunks = splitter.split_documents(
        documents
    )


    st.write(
        f"Total Chunks: {len(chunks)}"
    )


    # ------------------------
    # Embeddings
    # ------------------------
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # ------------------------
    # Groq LLM
    # ------------------------

    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
    )


    # ------------------------
    # Chroma DB
    # ------------------------
    vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectorstore"
    )

    

    st.success(
        "Vector database created!"
    )


    # Save vector_db in session
    st.session_state.vector_db = vector_db



# ------------------------
# Chat Section
# ------------------------

# ------------------------
# Chat Interface
# ------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



if "vector_db" in st.session_state:


    question = st.chat_input(
        "Ask something about your PDF..."
    )


    if question:


        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("user"):

            st.markdown(question)



        # Retrieve documents

        docs = st.session_state.vector_db.similarity_search(
            question,
            k=3
        )


        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )



        prompt = f"""

You are an AI document assistant.

Answer the question using the context.

If the answer is not available in the context,
use your general knowledge.

Context:

{context}


Question:

{question}


Answer:

"""



        with st.chat_message("assistant"):


            with st.spinner("Thinking..."):


                response = llm.invoke(prompt)


                st.markdown(response.content)



        # Store assistant message

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )


        