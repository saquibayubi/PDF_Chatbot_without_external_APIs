import streamlit as st
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import h5py

# Function to load FAISS index and metadata from HDF5 file
def load_faiss_index(filename):
    with h5py.File(filename, 'r') as f:
        index = faiss.deserialize_index(f['faiss_index'][()])
        metadata = [str(item, 'utf-8') for item in f['metadata']]
    return index, metadata

st.title("PDF Chatbot")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token to handle padding issues
llm_model.config.pad_token_id = tokenizer.eos_token_id  # Explicitly set pad_token_id

# Load FAISS index and metadata from HDF5 file
index, metadata = load_faiss_index('faiss_index.h5')
st.sidebar.write("We have these 3 PDFs \n\n 1. Google \n 2. Tesla \n 3. Uber \n\n\n\n\n")

# Query the vector store and generate insights
def query_insights(query, index, metadata, embedding_model, llm_model, tokenizer):
    query_embedding = embedding_model.encode([query])[0].astype('float32')
    D, I = index.search(np.array([query_embedding]), k=3)
    contexts = [metadata[i] for i in I[0]]
    
    # Provide context to the model without displaying it in the app
    context_texts = "\n\n".join(contexts)
    input_text = f"Context: {context_texts}\n\n Query: {query} \n\n Answer:" 

    # Tokenize the input text with padding
    inputs = tokenizer(input_text, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate output with attention mask
    output = llm_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,  # Lower temperature to reduce randomness
        top_k=50,  # Consider top k tokens
        top_p=0.9,  # Consider tokens with cumulative probability of 0.9
        repetition_penalty=2.0  # Add repetition penalty to reduce repeated phrases
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Display chat history
for speaker, text in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f'<div style="text-align: right; color: orange;"><b>User:</b> {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align: left; color: green;"><b>Bot:</b> {text}</div>', unsafe_allow_html=True)

# Input box at the bottom of the page
user_query = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_query:
        response = query_insights(user_query, index, metadata, embedding_model, llm_model, tokenizer)
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Bot", response))
        st.experimental_rerun()  # Rerun the script to update the UI

# Scroll to the bottom of the page to keep input box visible
st.write('<style>body {scroll-behavior: smooth;}</style>', unsafe_allow_html=True)
st.write('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)
