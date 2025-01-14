# 🤖 PDF Chat Assistant - RAG-based Document Q&A

A powerful and intuitive chat application that allows users to have interactive conversations with their PDF documents using state-of-the-art RAG (Retrieval-Augmented Generation) technology. Built with Streamlit and powered by OpenAI's language models.

## 🌟 Features

- **PDF Processing**: Upload and process multiple PDF documents simultaneously
- **Intelligent Text Chunking**: Advanced text splitting for optimal context preservation
- **Vector Search**: Utilizes FAISS for efficient similarity search
- **Conversational Memory**: Maintains context throughout the chat session
- **Token Usage Tracking**: Monitor OpenAI API usage in real-time
- **User-Friendly Interface**: Clean and intuitive Streamlit-based UI
- **Secure**: Environment-based API key management

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-chat-assistant.git
cd pdf-chat-assistant
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

### 📋 Requirements

```text
streamlit>=1.28.0
python-dotenv>=0.19.0
PyPDF2>=3.0.0
langchain>=0.0.300
openai>=1.0.0
faiss-cpu>=1.7.4
```

## 💡 Usage

1. **Launch the Application**
   - Start the application using `streamlit run app.py`
   - Access the web interface at `http://localhost:8501`

2. **Upload Documents**
   - Use the sidebar to upload one or more PDF files
   - Click "Process Documents" to initialize the system

3. **Start Chatting**
   - Type your questions in the input field
   - View responses and conversation history in real-time
   - Monitor token usage for each interaction

## 🏗️ Architecture

The application follows a modular architecture with the following components:

- **Document Processing**: Handles PDF text extraction and preprocessing
- **Text Chunking**: Splits text into manageable segments
- **Vector Store**: Creates and manages document embeddings
- **Conversation Chain**: Orchestrates the chat flow and context management
- **User Interface**: Streamlit-based frontend for user interactions

## 🛠️ Core Components

```python
def get_pdf_text(pdf_docs):
    # Extracts text from PDF documents
    
def get_text_chunks(text):
    # Splits text into smaller, manageable chunks
    
def get_vectorstore(text_chunks):
    # Creates embeddings and vector store
    
def get_conversation_chain(vectorstore):
    # Sets up the conversational chain
    
def handle_user_input(user_question):
    # Processes user input and generates responses
```

## 🔄 Workflow

1. **Document Upload**: User uploads PDF documents through the Streamlit interface
2. **Text Processing**: System extracts and chunks text from PDFs
3. **Embedding Creation**: Text chunks are converted to vector embeddings
4. **Query Processing**: User questions are processed against the vector store
5. **Response Generation**: OpenAI generates contextual responses
6. **Display**: Responses and chat history are displayed to the user

## 🎯 Future Enhancements

- [ ] Support for additional document formats (DOCX, TXT, etc.)
- [ ] Custom chunk size configuration
- [ ] Advanced error handling and retry mechanisms
- [ ] Chat history export functionality
- [ ] Multiple embedding model options
- [ ] User authentication and session management
- [ ] Docker containerization
- [ ] API endpoint creation


## 🙏 Acknowledgments

- OpenAI for providing the language model API
- Langchain for the conversation chain framework
- FAISS for vector similarity search
- Streamlit for the web interface framework