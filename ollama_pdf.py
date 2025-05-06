import PyPDF2
from flask import Flask, request, render_template, jsonify
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from ollama import chat
from ollama import ChatResponse
import os

# Step 1: Extract text from PDF
def extract_pdf_text(pdf_path):
    """Extract text from a PDF."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
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
        print("Quering OLLama")
        """Send a query to the Ollama model."""
        response: ChatResponse = chat(
            model= "gemma:2b", 
            messages=[  {
    'role': 'user',
    "content": f"Question: {question}\nContext: {context}"
  },
])
        print(response['message']['content'])

        if response:
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

# Step 4: Flask API
app = Flask(__name__)

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

pdf_file = "undefined"
chatbot = ""
@app.route("/query", methods=["POST"])
def query_pdf():
    try:
    
        query = request.form["query"]
        print(pdf_file)
        if not pdf_file or pdf_file == "undefined" or not query:
            return jsonify({"error": "PDF file and query are required."}), 400

        answer = chatbot.answer_question(query)
        # answer.replace("\xa0","<br>")

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pdf", methods=["POST"])
def upload_pdf():
    try:
        # Handle PDF upload
        global pdf_file, chatbot
        pdf_file = request.files["pdf"]

        if not pdf_file or pdf_file == "undefined" :
            return jsonify({"error": "PDF file and query are required."}), 400

        # Save the uploaded PDF
        pdf_path = "uploaded.pdf"
        pdf_file.save(pdf_path)

        # Process the query using RAG pipeline
        # Load the PDF and initialize the chatbot
        pdf_text = extract_pdf_text(pdf_path)  # Replace with your PDF file path
        knowledge_base = create_knowledge_base(pdf_text)
        chatbot = PDFChatbot(knowledge_base)

        # Cleanup the uploaded file
        os.remove(pdf_path)

        return jsonify({"file": pdf_file.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)