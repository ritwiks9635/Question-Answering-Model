# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the necessary packages
RUN pip install -r requirements.txt

# Expose the port Gradio will run on
EXPOSE 7860

# Set the environment variables for the API keys (can pass values at runtime)
ENV PINECONE_API_KEY=$PINECONE_API_KEY
ENV COHERE_API_KEY=$COHERE_API_KEY

# Run the Python script
CMD ["python", "Question_Answering.py"]
