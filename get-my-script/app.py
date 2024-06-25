import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

# Load the CSV file
url = 'https://raw.githubusercontent.com/gonzalophd/getscript/main/get-my-script/script_DB_eng.csv'

@st.cache_data(ttl=86400)  # Cache for one day
def load_original_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to download the database from Github.")
        return None

df = load_original_data(url)

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

@st.cache_resource(ttl=86400)  # Cache the model for one day
def load_model(model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    return model

@st.cache_data(ttl=86400)  # Cache embeddings for one day
def compute_embeddings(descriptions, model):
    embeddings = model.encode(descriptions, batch_size=32, show_progress_bar=False)  # Adjust batch_size based on your available memory
    return embeddings

# Load model as a part of session state to ensure it's only loaded once
if 'model' not in st.session_state:
    with st.spinner('Loading model...'):
        st.session_state.model = load_model()
        st.success('Model loaded.')

# Pre-compute embeddings if necessary
if 'embeddings' not in st.session_state and df is not None and 'Description' in df.columns:
    with st.spinner('Computing embeddings...'):
        descriptions = df['Description'].tolist()
        embeddings = compute_embeddings(descriptions, st.session_state.model)
        st.session_state.embeddings = embeddings
        st.success('Embeddings computed.')
        st.write(f"Number of embeddings computed: {len(embeddings)}")

# Streamlit app interface
st.title("Get my Script!")
st.write("This app helps you find the script that best matches your needs.")

user_description = st.text_area("Enter a description:", "")
threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.3)
recommendation = st.checkbox("Give me recommendations if there are no exact matches.")

if st.button("Get My Script!"):
    if user_description:
        if 'model' not in st.session_state or st.session_state.model is None:
            st.write("The model is still loading. Please wait a moment and try again.")
        elif 'embeddings' not in st.session_state or st.session_state.embeddings is None:
            st.write("The embeddings are still loading. Please wait a moment and try again.")
        else:
            def find_best_matches(description, threshold=0.3, recommendation=False):
                start_time = time.time()
                user_embedding = st.session_state.model.encode([description])  # Use already loaded model
                cosine_similarities = cosine_similarity([user_embedding], st.session_state.embeddings)[0]
                sorted_indices = cosine_similarities.argsort()[::-1]
                matches = [(df.iloc[idx]['Script Name'], df.iloc[idx]['Path'], cosine_similarities[idx]) for idx in sorted_indices if cosine_similarities[idx] >= threshold]
                if not matches and recommendation:
                    matches = [(df.iloc[idx]['Script Name'], df.iloc[idx]['Path'], cosine_similarities[idx]) for idx in sorted_indices[:3]]
                end_time = time.time()
                return matches, end_time - start_time

            matches, time_taken = find_best_matches(user_description, threshold, recommendation)
            if not matches:
                st.write("Sorry, no matches found.")
            else:
                results_df = pd.DataFrame(matches, columns=["Script Name", "Path", "Similarity Score"])
                st.write("### Best Matches Found:")
                st.dataframe(results_df)
                st.write(f"**Time taken:** {time_taken:.4f} seconds")
    else:
        st.write("Please enter a description.")
