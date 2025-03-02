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
df = pd.read_csv("akc-data-latest.csv")
df.rename(columns={df.columns[0]: 'Breed'}, inplace=True)
json_data = df.to_json(orient='records')
# article_chunks = str(json_data)

# Convert JSON table to text format
def convert_json_to_text(data):
    text_data = "\n".join([f"{row}" for row in data])
    return text_data

doc_text = convert_json_to_text(json_data)

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = [Document(page_content=text) for text in text_splitter.split_text(doc_text)]

# Use Hugging Face Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index with LangChain wrapper
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()

# Load Zephyr-7B model from Hugging Face
llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha", model_kwargs={"temperature": 0.1, "top_p": 0.5},huggingfacehub_api_token ="")

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Function to ask questions
def ask_question(query):
    return qa_chain.run(query)

# Example query
query = "What breeds are known for being both protective and good with families?"
response = ask_question(query)
print(response)
