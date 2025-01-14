import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create a conversation chain using the vector store."""
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    """Process user input and generate response."""
    if st.session_state.conversation:
        with get_openai_callback() as cb:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"Human: {message.content}")
                else:
                    st.write(f"Assistant: {message.content}")
            
            # Display token usage
            st.write(f"\nToken Usage: {cb.total_tokens} tokens")

def main():
    # Configure Streamlit page
    st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")
    st.header("Chat with Multiple PDFs 💬")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Create sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", 
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(f"Created {len(text_chunks)} text chunks")
                
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write("Created vector store")
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully!")

    # Create main chat interface
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()