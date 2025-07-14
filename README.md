🔎 CloseEnough — Finds What’s Closest to What You Meant

CloseEnough is an interactive Streamlit app that uses Sentence Transformers and cosine similarity to find the most relevant sentence from a 
predefined list of sentences based on your query.



📋 FEATURES:

✅ Enter a query and find the closest matching sentence.


✅ Displays the most similar sentence and the similarity score.

✅ Uses powerful embeddings from sentence-transformers.

✅ Easy-to-use web interface powered by Streamlit.



🚀 HOW TO RUN:


1️⃣ Clone the repository

# Replace 'username' and 'repository-name' with your actual values
git clone https://github.com/username/repository-name.git

# Navigate to the project directory
cd repository-name


2️⃣ INSTALL DEPENDENCIES:
 

4️⃣ RUN THE APP:

streamlit run search_app.py

Then open the URL (e.g., http://localhost:8501) in your browser.



📄 FILES:
File	        Description
search_app.py	Streamlit app code
sentences.txt	List of sentences for matching



🛠️ TECH STACK:

🐍 Python

📝 Streamlit

🧠 Sentence Transformers (all-MiniLM-L6-v2)

📐 scikit-learn

🔢 NumPy

📷 Sample Output

✨ EXAMPLE:

Query:

How do machines learn from data?

Best Match:

Machine learning is a subset of AI that enables systems to learn from data.

Similarity Score:

0.681


📜 LICENSE:

MIT License.

