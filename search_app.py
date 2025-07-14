import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Read sentences from file
def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

sentences = load_sentences("sentences.txt")
embeddings = model.encode(sentences)

def find_best_match(query, embeddings, sentences):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    best_idx = np.argmax(similarities)
    return sentences[best_idx], similarities[best_idx]

# Streamlit UI
st.title("🔎 CloseEnough — Finds What’s Closest to What You Meant")
st.write("What’s on your mind? Type it here and I’ll find the closest match!")

query = st.text_input("Enter your query:")

if query:
    match, score = find_best_match(query, embeddings, sentences)
    st.write(f"*Best Match:* {match}")
    st.write(f"*Similarity Score:* {score:.3f}")