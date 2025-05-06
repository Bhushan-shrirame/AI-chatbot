import streamlit as st
# import PyPDF2
import pdfplumber
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from ollama import chat
from ollama import ChatResponse
import os

# Step 1: Extract text from PDF
def extract_pdf_text_temp(pdf_path):
    """Extract text from a PDF."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text
def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# Step 2: Create a knowledge base
def create_knowledge_base(pdf_text):
    """Split the text and create a vectorized knowledge base."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(pdf_text)
    
    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

# Step 3: Create the chatbot using Ollama
class PDFChatbot:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def query_ollama(self, question, context):
        # print("Quering OLLama")
        """Ollama model"""
        response: ChatResponse = chat(
            model= "gemma:2b", 
            messages=[  {
    'role': 'user',
    "content": f"Question: {question}\nContext: {context}"
  },
])
        print(response['message']['content'])


        if response:
            if "context does not provide any information" in response['message']['content']:
                return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
            return response['message']['content']
        return "Sorry, I couldn’t connect to the local model."

    def answer_question(self, question):
        # Retrieve the most relevant chunk from the knowledge base
        print("Looking for similarities")
        results = self.knowledge_base.similarity_search(question, k=1)
        if not results:
            return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
        
        context = results[0].page_content
        return self.query_ollama(question, context)


# App title
st.set_page_config(page_title="Bhushan Shrirame Assignment", layout="wide")
# st.set_page_config(page_title="Page Title", layout="wide")

# Hugging Face Credentials
with st.sidebar:
    st.title('AI Chatbot Assignment')
    st.write('~ Bhushan Shrirame Assignment ')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.write("You uploaded a file of size", len(bytes_data), "bytes")

         # Add a submit button
        if st.button("Submit"):
            # Process the uploaded file here
            st.write("Processing the file...")
            st.session_state.pdf_file = uploaded_file.name
            st.session_state.knowledge_base = create_knowledge_base(extract_pdf_text(st.session_state.pdf_file))
            st.session_state.chatbot = PDFChatbot(st.session_state.knowledge_base)
            st.write("knowledge Base Created!")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input):
    answer = st.session_state.chatbot.answer_question(prompt_input)
    return answer

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)



# st.markdown("""
#     <style>
#         .reportview-container {
#             margin-top: -2em;
#         }
#         #MainMenu {visibility: hidden;}
#         .stDeployButton {display:none;}
#         footer {visibility: hidden;}
#         #stDecoration {display:none;}
#     </style>
# """, unsafe_allow_html=True)bye