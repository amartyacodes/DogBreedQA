# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the required files
COPY requirements.txt .
COPY akc-data-latest.csv .
COPY gradio_faiss_qa.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Gradio uses
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "gradio_faiss_qa.py"]
