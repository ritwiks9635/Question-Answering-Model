#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ritwiks9635/Question-Answering-Model/blob/main/Question_Answering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


# Install necessary libraries
get_ipython().system('pip install pinecone-client cohere streamlit gradio langchain')
get_ipython().system('pip install sentence-transformers # for generating embeddings')


# In[ ]:


get_ipython().system('pip install pymupdf')


# In[25]:


import pinecone
import cohere
import os

# Set your API keys here
PINECONE_API_KEY = os.getenv('API_KEY')
COHERE_API_KEY = os.getenv('API_KEY')


# In[4]:


import fitz

# Function to extract text from a PDF file using PyMuPDF
def extract_text_from_pdf_with_pymupdf(pdf_file):
    pdf_reader = fitz.open(pdf_file)
    text = ''
    for page_num in range(13, len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text += page.get_text("text")  # Extracts the text from each page
    return text

# Test with a sample PDF file
pdf_path = '/content/ML Notes.pdf'  # Replace with your PDF path
document_text = extract_text_from_pdf_with_pymupdf(pdf_path)

print(document_text[:500].strip())  # Print the first 500 characters to verify


# In[ ]:


from sentence_transformers import SentenceTransformer

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the extracted document text
document_embeddings = model.encode(document_text)

print(f"Document Embeddings Shape: {document_embeddings.shape}")


# In[6]:


# Create Pinecone index
index_name = 'qa-bot'

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(PINECONE_API_KEY, environment='us-west1-gcp')

# Check if index exists
pc.create_index(
    name=index_name,
    dimension=len(document_embeddings), # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) )
index = pc.Index(index_name)


# In[7]:


# Function to split the document into smaller chunks
def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Example: Split the document text into chunks of 300 words each
chunks = split_text_into_chunks(document_text, chunk_size=300)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[1][:500]}")  # Print first 500 characters of the first chunk


# In[8]:


# Function to store document chunks and their embeddings in Pinecone
def store_chunks_in_pinecone(chunks):
    for i, chunk in enumerate(chunks):
        chunk_embedding = model.encode(chunk)

        chunk_id = f'doc_chunk_{i}'  # Unique ID for each chunk
        index.upsert([(chunk_id, chunk_embedding.tolist())])

# Store the chunks in Pinecone
store_chunks_in_pinecone(chunks)
print("Document chunks and embeddings stored in Pinecone.")


# In[9]:


# Function to generate embeddings for a query
def generate_query_embedding(query):
    query_embedding = model.encode(query)
    return query_embedding

# Example query
user_query_1 = "Why Data Cleansing Is Important In Data Analysis?"
query_embedding = generate_query_embedding(user_query_1)
print(f"Query Embedding Shape: {query_embedding.shape}")


# In[10]:


def retrieve_relevant_documents(query_embedding, top_k=3):
    # Ensure query embedding is a list of floats
    query_embedding = query_embedding.astype(float).tolist()

    # Search the Pinecone index for the top_k most similar document embeddings
    result = index.query(
       # namespace="example-namespace",  # Replace with your actual namespace
        vector=query_embedding,
        top_k=top_k,
        include_values=True
    )
    return result


# In[11]:


def answer_question(user_query, top_k=3):
    # 1. Generate query embedding
    query_embedding = generate_query_embedding(user_query)

    # 2. Retrieve the most relevant chunks based on the query
    retrieval_result = retrieve_relevant_documents(query_embedding, top_k)

    # 3. Fetch the text of the top retrieved chunk
    top_chunk_id = retrieval_result['matches'][0]['id']  # Get the top chunk ID
    top_chunk_index = int(top_chunk_id.split('_')[-1])  # Extract chunk index from ID
    retrieved_chunk = chunks[top_chunk_index]  # Fetch the chunk from the list of chunks

    return retrieved_chunk

# Example: Ask a new question and get a chunk-level answer
new_question = "What is Deeplearning?"
chunk_answer = answer_question(new_question)
print(f"Answer: {chunk_answer}")  # Print first 500 characters of the chunk


# In[15]:


co = cohere.Client(COHERE_API_KEY)


# In[13]:


def generate_answer_with_cohere(context, question):
    # Combine the context (retrieved chunk) and the question to form the prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    # Use Cohere's generate function to generate the answer
    response = co.generate(
        model='command-xlarge-nightly',  # You can choose a model here
        prompt=prompt,
        max_tokens=100,  # Adjust token limit as needed
        temperature=0.7  # Adjust temperature for creativity
    )

    # Extract the generated answer
    generated_answer = response.generations[0].text.strip()
    return generated_answer


# In[16]:


def answer_question(user_query, top_k=3):
    # 1. Generate the query embedding
    query_embedding = generate_query_embedding(user_query)

    # 2. Retrieve the most relevant chunks from Pinecone
    retrieval_result = retrieve_relevant_documents(query_embedding, top_k)
    top_chunk_id = retrieval_result['matches'][0]['id']  # Get the top chunk ID
    top_chunk_index = int(top_chunk_id.split('_')[-1])  # Extract chunk index from ID
    retrieved_chunk = chunks[top_chunk_index]  # Fetch the chunk from the list of chunks

    # 3. Generate the final answer using Cohere
    generated_answer = generate_answer_with_cohere(retrieved_chunk, user_query)

    return generated_answer

# Example: Ask a new question and get a generated answer
new_question = "What is machine learning?"
final_answer = answer_question(new_question)
print(f"Final Answer: {final_answer}")


# In[18]:


new_question = "What is Multiple Linear Regression?"
final_answer = answer_question(new_question)
print(f"Final Answer: {final_answer}")


# In[21]:


import gradio as gr

# Function to extract text from a PDF file using PyMuPDF
def extract_text_from_pdf_with_pymupdf(pdf_file):
    pdf_reader = fitz.open(pdf_file)
    text = ''
    for page_num in range(13, len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text += page.get_text("text")  # Extracts the text from each page
    return text
# Function to handle question and answer process
def qa_system(pdf_file, user_question):
    # 1. Extract the document text from the uploaded PDF
    document_text = extract_text_from_pdf_with_pymupdf(pdf_file)

    # 2. Split the document into chunks, generate embeddings, and store them in Pinecone
    chunks = split_text_into_chunks(document_text, chunk_size=300)
    store_chunks_in_pinecone(chunks)

    # 3. Retrieve and generate the answer based on the user's question
    answer = answer_question(user_question)

    return answer

# Gradio interface for document upload and question answering
def create_gradio_interface():
    interface = gr.Interface(
        fn=qa_system,  # The function that processes the PDF and answers the question
        inputs=[
            gr.components.File(label="Upload PDF Document"),  # PDF file upload
            gr.components.Textbox(label="Ask a question")  # User question input
        ],
        outputs=gr.components.Textbox(label="Answer")  # Answer output
    )
    return interface

# Launch the Gradio app
gradio_interface = create_gradio_interface()
gradio_interface.launch()


# In[22]:


gradio_interface.close()


# In[ ]:




