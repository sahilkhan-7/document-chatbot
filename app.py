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
from langchain.prompts import ChatPromptTemplate
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
    
    # Define class attributes properly
    temperature: float = 0.3
    max_tokens: int = 500
    
    def __init__(self, temperature: float = 0.3, max_tokens: int = 500):
        """Initialize the GroqLLM with temperature and max_tokens parameters."""
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generates a response from the Groq LLM based on the provided prompt.

        Args:
            prompt (str): The complete prompt with context and question.
            stop (Optional[List[str]]): List of stop tokens (not used here).

        Returns:
            str: The generated response from the LLM.
        """
        client = Groq(api_key=GROQ_API_KEY)
        
        # Define system prompt for better responses
        messages = [
            {"role": "system", "content": 
             """You are a knowledgeable assistant that answers questions based solely on the provided context.
             Focus on giving precise, factual answers that directly address the question.
             Use bullet points for clarity when appropriate.
             If the information isn't in the context, admit you don't know rather than making up answers.
             Format your responses in a clean, readable manner."""}
        ]

        # Append the user prompt (which already contains context from LangChain)
        messages.append({"role": "user", "content": prompt})

        try:
            # Generate a response using Groq LLM
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-specdec",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response = chat_completion.choices[0].message.content.strip()
            return response
        except Exception as e:
            return f"Error: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Returns the identifying parameters for the custom LLM."""
        return {"name": "GroqLLM", "temperature": self.temperature, "max_tokens": self.max_tokens}

    @property
    def _llm_type(self) -> str:
        """Returns the type of the LLM."""
        return "groq"


def load_documents(uploaded_files):
    """Loads and processes uploaded files into a list of document objects."""
    documents = []
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Track progress
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_filepath)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_filepath)
        else:
            loader = UnstructuredFileLoader(temp_filepath)

        documents.extend(loader.load())
        os.remove(temp_filepath)  # Cleanup
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    return documents


def split_documents(documents):
    """Splits the loaded documents into smaller chunks for better context handling."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def create_vector_store(chunks):
    """Creates a FAISS vector store from document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def create_custom_prompt():
    """Creates a custom prompt template for better RAG responses."""
    template = """
    Answer the user's question using ONLY the following context and your general knowledge about how to interpret such information.
    If the answer cannot be found in the context, respond with "I don't have enough information to answer that question."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    YOUR ANSWER:
    """
    return ChatPromptTemplate.from_template(template)


def create_conversational_retrieval_chain(vector_store):
    """Creates an enhanced conversational retrieval chain."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Using a custom prompt for more accurate responses
    custom_prompt = create_custom_prompt()
    
    return ConversationalRetrievalChain.from_llm(
        llm=GroqLLM(temperature=0.2, max_tokens=600),
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 4, "fetch_k": 8}
        ),
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        verbose=False
    )


def handle_user_input(user_question):
    """Handles user input and generates a response using the conversation chain."""
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.warning("Please process the documents first.")
        return
    
    with st.spinner("Thinking..."):
        response = st.session_state.conversation({'question': user_question})
    
    # Extract only the answer
    response_text = response.get("answer", "No response generated.")
    
    # Update the chat history for display purposes
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=response_text))


def main():
    """
    Main function to set up the Streamlit interface and handle user interactions.
    """
    st.set_page_config(page_title="Document Chat Assistant", layout="wide", page_icon="📚")
    st.header("📚 Chat with Your Documents")
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Set up the sidebar for document upload and processing
    with st.sidebar:
        st.subheader("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your files here", 
            type=['pdf', 'docx', 'txt'], 
            accept_multiple_files=True
        )
        
        # Process Files Button
        if st.button("Process Documents", disabled=st.session_state.processing):
            if not uploaded_files:
                st.error("Please upload at least one document.")
            else:
                st.session_state.processing = True
                with st.spinner("Processing documents..."):
                    try:
                        documents = load_documents(uploaded_files)
                        st.write(f"✅ Loaded {len(documents)} document sections")
                        
                        chunks = split_documents(documents)
                        st.write(f"✅ Created {len(chunks)} text chunks")
                        
                        vector_store = create_vector_store(chunks)
                        st.session_state.vector_store = vector_store  # Store vector_store in session state
                        st.write("✅ Created vector store")
                        
                        st.session_state.conversation = create_conversational_retrieval_chain(vector_store)
                        st.success("Documents processed successfully! ✅")
                        st.session_state.processing = False
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.session_state.processing = False
        
        # Clear Chat Button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            if st.session_state.conversation:
                st.session_state.conversation.memory.clear()
            st.success("Chat history cleared! ✅")
        
        # Clear Database Button
        if st.button("Clear Database"):
            st.session_state.vector_store = None
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.success("Vector database cleared! ✅")
        
        # Simple document information
        if uploaded_files:
            st.subheader("📊 Documents")
            for uploaded_file in uploaded_files:
                st.write(f"**{uploaded_file.name}** ({round(uploaded_file.size / 1024, 2)} KB)")
        
        # Database status indicator
        if st.session_state.vector_store is not None:
            st.success("Database: Active ✓")
        else:
            st.warning("Database: Inactive ✗")

    # Chat container with messages
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(f"#### 🧑 **You:**\n{message.content}")
                st.markdown("---")
            elif isinstance(message, AIMessage):
                st.markdown(f"#### 🤖 **Assistant:**\n{message.content}")
                st.markdown("---")
        
        # If no messages, show a simple greeting
        if not st.session_state.chat_history and st.session_state.vector_store is None:
            st.info("Upload your documents in the sidebar and hit 'Process Documents' to get started.")
    
    # Function to handle form submission
    def process_input():
        if st.session_state.user_input and st.session_state.conversation:
            handle_user_input(st.session_state.user_input)
            # Clear the input after processing
            st.session_state.user_input = ""
    
    # Create a container to hold the text input
    input_container = st.container()
    
    # Add the text input for questions
    with input_container:
        # Use a key press detector to handle Enter key
        user_question = st.text_input(
            "Ask a question about your documents:",
            key="user_input",
            placeholder="Type your question and press Enter...",
            on_change=process_input  # Process on change (Enter key press)
        )


# Run the App
if __name__ == '__main__':
    main()