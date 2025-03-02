import gradio as gr
import json
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

# Load and process dataset
df = pd.read_csv("akc-data-latest.csv")
df.rename(columns={df.columns[0]: 'Breed'}, inplace=True)
json_data = df.to_json(orient='records')

def convert_json_to_text(data):
    text_data = "\n".join([f"{row}" for row in data])
    return text_data

doc_text = convert_json_to_text(json_data)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
documents = [Document(page_content=text) for text in text_splitter.split_text(doc_text)]

# Initialize FAISS index with LangChain wrapper
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()

# Load Zephyr-7B model from Hugging Face
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-alpha", 
    model_kwargs={"temperature": 0.1, "top_p": 0.5},
    huggingfacehub_api_token="" #Your API 
)

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(query):
    return qa_chain.run(query)

# Gradio interface
demo = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=gr.Textbox(label="Answer"),
    title="Dog Breed Q&A",
    description="Ask a question about dog breeds and get an AI-generated answer."
)

demo.launch()
