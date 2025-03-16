import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from typing import Optional, List, Mapping, Any
from dotenv import load_dotenv

# Load environment variables (Groq API Key)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class GroqLLM(LLM):
    """
    Custom LLM class that extends LangChain's base LLM.
    Uses the Groq API to generate responses based on user input and chat history.
    """

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generates a response from the Groq LLM based on the provided prompt and chat history.

        Args:
            prompt (str): The user input/question.
            stop (Optional[List[str]]): List of stop tokens (not used here).

        Returns:
            str: The generated response from the LLM.
        """
        client = Groq(api_key=GROQ_API_KEY)
        chat_history = st.session_state.get("chat_history", [])

        # Define system prompt for structured and concise answers
        messages = [
            {"role": "system", "content": 
             "You are a helpful assistant. Provide structured and concise responses in bullet points or numbered lists where applicable. Avoid long paragraphs and redundant information."}
        ]

        # Add previous conversation history to the context
        for turn in chat_history:
            if isinstance(turn, HumanMessage):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AIMessage):
                messages.append({"role": "assistant", "content": turn.content})

        # Append the latest user input
        messages.append({"role": "user", "content": prompt})

        try:
            # Generate a response using Groq LLM
            chat_completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=0.3, 
                max_tokens=300  
            )
            response = chat_completion.choices[0].message.content.strip()

            # Save the conversation to session state
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=response))

            return response
        except Exception as e:
            return f"Error: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Returns the identifying parameters for the custom LLM.

        Returns:
            dict: A dictionary containing the LLM name.
        """
        return {"name": "GroqLLM"}

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of the LLM.

        Returns:
            str: LLM type.
        """
        return "groq"


def load_documents(uploaded_files):
    """
    Loads and processes uploaded files into a list of document objects.

    Args:
        uploaded_files (list): List of uploaded files from Streamlit file uploader.

    Returns:
        list: A list of processed document objects.
    """
    documents = []
    for uploaded_file in uploaded_files:
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file to a temporary directory
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Identify the file type and load using appropriate loader
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_filepath)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_filepath)
        else:
            loader = UnstructuredFileLoader(temp_filepath)

        # Load the document content
        documents.extend(loader.load())

        # Remove temporary file after processing
        os.remove(temp_filepath)
    
    return documents


def split_documents(documents):
    """
    Splits the loaded documents into smaller chunks for better context handling.

    Args:
        documents (list): List of document objects.

    Returns:
        list: List of split document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    """
    Converts document chunks into embeddings and stores them in a FAISS vector store.

    Args:
        chunks (list): List of document chunks.

    Returns:
        FAISS: FAISS vector store containing document embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def create_conversational_retrieval_chain(vector_store):
    """
    Creates a conversational retrieval chain using LangChain and FAISS.

    Args:
        vector_store (FAISS): FAISS vector store containing document embeddings.

    Returns:
        ConversationalRetrievalChain: Retrieval chain for document-based Q&A.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=GroqLLM(),
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return qa_chain

# Handle user input
def handle_user_input(user_question):
    """
    Handles user input and generates a response using the conversation chain.

    Args:
        user_question (str): User's question.

    Returns:
        None
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

# MAIN FUNCTION
def main():
    """
    Main function to set up the Streamlit interface and handle user interactions.
    """
    st.set_page_config(page_title="Chat with Multiple Documents", layout="wide", page_icon=":books:")
    st.header("Chat with Documents using LangChain and Groq")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Upload your Documents")
        uploaded_files = st.file_uploader("Upload your files here", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

        # Process Files Button
        if st.button("Process Documents"):
            if not uploaded_files:
                st.error("Please upload at least one document.")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        documents = load_documents(uploaded_files)
                        chunks = split_documents(documents)
                        st.write(f"✅ Created {len(chunks)} text chunks")

                        vector_store = create_vector_store(chunks)
                        st.write("✅ Created vector store")

                        st.session_state.conversation = create_conversational_retrieval_chain(vector_store)
                        st.success("Documents processed successfully! ✅")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        if st.session_state.conversation:
            st.session_state.conversation.memory.clear()
        st.success("Chat history cleared! ✅")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please process the documents first.")
        else:
            handle_user_input(user_question)

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.write(f"**User:** {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"**Assistant:** {message.content}")

# Run the App
if __name__ == '__main__':
    main()
