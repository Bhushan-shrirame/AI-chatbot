from flask import Flask, request, render_template, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import os
import ollama

# Initialize Flask app
app = Flask(__name__)

# Initialize models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="t5-small")
grammar_corrector = pipeline("text2text-generation", model="t5-small")  # T5 for grammar correction

def ollama_use():
    response = ollama.generate(model='gemma:2b',
    prompt='what is a qubit?')
    print(response['response'])

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Step 2: Split text into chunks
def split_into_chunks(text, chunk_size=512):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Step 3: Create an embedding index
def create_index(chunks):
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    return chunks, chunk_embeddings

# Step 4: Retrieve the most relevant chunk
def retrieve_relevant_chunk(query, chunks, chunk_embeddings):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
    best_chunk_idx = torch.argmax(similarities).item()
    return chunks[best_chunk_idx]

# Step 5: Summarize the chunk
def summarize_chunk(chunk, query):
    input_text = f"Extract relevant information to answer the question: {query}\n\n{chunk}"
    summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Step 6: Correct grammatical errors
def correct_grammar(text):
    corrected = grammar_corrector(f"fix grammar: {text}", max_length=100, do_sample=False)
    return corrected[0]["generated_text"]

# Main function to answer query
def answer_query_with_rag(pdf_path, query):
    # Extract and preprocess text from the PDF
    text = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(text)

    # Create searchable index
    chunks, chunk_embeddings = create_index(chunks)

    # Retrieve the most relevant chunk
    relevant_chunk = retrieve_relevant_chunk(query, chunks, chunk_embeddings)

    # Summarize the chunk
    summary = summarize_chunk(relevant_chunk, query)

    # Correct grammar
    corrected_answer = correct_grammar(summary)

    return corrected_answer

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query_pdf():
    ollama_use()
    try:
        # Handle PDF upload
        pdf_file = request.files["pdf"]
        query = request.form["query"]

        if not pdf_file or not query:
            return jsonify({"error": "PDF file and query are required."}), 400

        # Save the uploaded PDF
        pdf_path = "uploaded.pdf"
        pdf_file.save(pdf_path)

        # Process the query using RAG pipeline
        answer = answer_query_with_rag(pdf_path, query)

        # Cleanup the uploaded file
        os.remove(pdf_path)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask, request, render_template, jsonify
# import pdfplumber
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Load Falcon-7B model
# def load_falcon_model():
#     model_name = "tiiuae/falcon-7b-instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     return tokenizer, model

# # Extract text from the uploaded PDF
# def extract_text_from_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"
#     return text

# # Generate a response from the Falcon-7B model
# def generate_response(tokenizer, model, context, query):
#     prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=300,
#         do_sample=True,
#         temperature=0.7
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Load the model and tokenizer
# tokenizer, model = load_falcon_model()

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/query", methods=["POST"])
# def query_pdf():
#     try:
#         # Handle PDF upload
#         pdf_file = request.files.get("pdf")
#         query = request.form.get("query")

#         if not pdf_file or not query:
#             return jsonify({"error": "PDF file and query are required."}), 400

#         # Save PDF locally
#         pdf_path = "uploaded.pdf"
#         pdf_file.save(pdf_path)

#         # Extract text from the PDF
#         context = extract_text_from_pdf(pdf_path)

#         # Generate response
#         response = generate_response(tokenizer, model, context[:2000], query)  # Limit context to 2000 characters

#         # Cleanup
#         os.remove(pdf_path)

#         return jsonify({"answer": response})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


#******************************

# from flask import Flask, request, render_template, jsonify
# import pdfplumber
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# app = Flask(__name__)

# # Load Flan-T5 Small
# flan_model_name = "google/flan-t5-small"
# flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
# flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"
#     return text

# # Generate response using Flan-T5 Small
# def generate_flan_response(context, query):
#     prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
#     inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True)
#     outputs = flan_model.generate(inputs["input_ids"], max_length=200)
#     return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/query", methods=["POST"])
# def query_pdf():
#     try:
#         pdf_file = request.files.get("pdf")
#         query = request.form.get("query")

#         if not pdf_file or not query:
#             return jsonify({"error": "PDF file and query are required."}), 400

#         # Save PDF locally
#         pdf_path = "uploaded.pdf"
#         pdf_file.save(pdf_path)

#         # Extract text from the PDF
#         context = extract_text_from_pdf(pdf_path)[:2000]  # Limit context to 2000 characters

#         # Generate response
#         response = generate_flan_response(context, query)
#         return jsonify({"answer": response})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, render_template, jsonify
# import pdfplumber
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = Flask(__name__)

# # Load Falcon-7B-Instruct on CPU
# falcon_model_name = "tiiuae/falcon-7b-instruct"
# falcon_tokenizer = AutoTokenizer.from_pretrained(falcon_model_name)
# falcon_model = AutoModelForCausalLM.from_pretrained(
#     falcon_model_name,
#     torch_dtype=torch.float32,  # Use 32-bit precision for compatibility
#     device_map=None  # Force CPU usage
# )

# # Preload PDF file
# PDF_PATH = "input.pdf"  # Replace with the path to your preloaded PDF file


# def extract_text_from_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         text = ""
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"
#     return text


# # Extract text from the preloaded PDF
# pdf_context = extract_text_from_pdf(PDF_PATH)


# # Generate response using Falcon-7B-Instruct
# def generate_falcon_response(context, query):
#     prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
#     inputs = falcon_tokenizer(prompt, return_tensors="pt", truncation=True)
#     inputs = inputs.to("cpu")  # Explicitly move to CPU
#     outputs = falcon_model.generate(
#         inputs["input_ids"],
#         max_new_tokens=200,  # Limit output length to 200 tokens
#         temperature=0.7
#     )
#     return falcon_tokenizer.decode(outputs[0], skip_special_tokens=True)


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/query", methods=["POST"])
# def query_pdf():
#     try:
#         query = request.form.get("query")

#         if not query:
#             return jsonify({"error": "Query is required."}), 400

#         # Generate response
#         response = generate_falcon_response(pdf_context[:2000], query)  # Limit context to 2000 characters
#         return jsonify({"answer": response})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True)
