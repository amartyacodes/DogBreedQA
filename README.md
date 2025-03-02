# DogBreedQA
Repository to Gradio app for Question- Answering using CSV data with RAG Methodology

**Note: Please use the personal Huggingface token inside the code to run the model. It couldn't be added due to privacy issues.**



## Solution
1. The initial CSV file was read and the dog breed columns were identified.
2. The csv data was converted into a Json data structure and then finally converted into text document to be fed to the RAG Pipeline
3. The local embeddings were used instead to embed the text of Open AI embeddings (as using APIs incur costs and we want the project to be open-source)
4. This text embeddings were stored in a Vector Database and retrieved using Facebook AI Similarity Search i.e. FAISS
5. Then Zephyr 7B was used to generate the answer using the query. Again instead of this GPT 3.5/4 could have been used for better performance

**In order to run the main file**
Run: ```python main.py```

## Running the Gradio App
The Gradio Application will take in the question about the dataset and give a string output. In order to run the Gradio app 
Run: ```python gradio_faiss_qa.py```
<img width="1241" alt="Screenshot 2025-03-02 183046" src="https://github.com/user-attachments/assets/44ac7c2e-0cd7-4e13-96e1-dbdac5c503a1" />


**The docker file has also been given in the repository, in order to create the Docker Package**
