import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import random

# Load the CSV file
# url = 'https://raw.githubusercontent.com/gonzalophd/getscript/main/get-my-script/script_DB_eng.csv'
url = 'https://raw.githubusercontent.com/gonzalophd/getscript/main/get-my-script/script_DB_eng.feather'

@st.cache_data(ttl=86400)  # Cache for one day
def load_original_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_feather(StringIO(response.text))
        # df = pd.read_csv(StringIO(response.text))
        # Convert all relevant columns to lowercase
        df['Description'] = df['Description'].str.lower()
        df['Category/Software'] = df['Category/Software'].str.lower()
        df['Keywords'] = df['Keywords'].str.lower()
        return df
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
def compute_embeddings(combined_texts, _model):
    embeddings = _model.encode(combined_texts, batch_size=32, show_progress_bar=False)  # Adjust batch_size based on your available memory
    return np.array(embeddings)

# Combine relevant columns into a single text for each row
df['Combined'] = df['Description'] + ' ' + df['Category/Software']*2 + ' ' + df['Keywords']*2

# Load model as a part of session state to ensure it's only loaded once
if 'model' not in st.session_state:
    with st.spinner('Loading model...'):
        st.session_state.model = load_model()

# Pre-compute embeddings if necessary
if 'embeddings' not in st.session_state and df is not None and 'Combined' in df.columns:
    with st.spinner('Computing embeddings...'):
        combined_texts = df['Combined'].tolist()
        embeddings = compute_embeddings(combined_texts, st.session_state.model)
        st.session_state.embeddings = embeddings


# Streamlit app interface
st.title("Get my Script!")
st.write("This app helps you find the script that best matches your needs.")

# Initialize session state for user description and recommendation
if 'user_description' not in st.session_state:
    st.session_state.user_description = ""
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = False

# User inputs
st.session_state.user_description = st.text_area("Enter a description:", st.session_state.user_description).lower()
st.session_state.recommendation = st.checkbox("Give me recommendations if there are no exact matches.", st.session_state.recommendation)
threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

if st.button("Get My Script!"):
    if st.session_state.user_description:
        if 'model' not in st.session_state or st.session_state.model is None:
            st.write("The model is still loading. Please wait a moment and try again.")
        elif 'embeddings' not in st.session_state or st.session_state.embeddings is None:
            st.write("The embeddings are still loading. Please wait a moment and try again.")
        else:
            def find_best_matches(description, threshold=0.3, recommendation=False):
                start_time = time.time()
                user_embedding = st.session_state.model.encode([description])
                user_embedding = np.array(user_embedding)  # Ensure it's a numpy array
                cosine_similarities = cosine_similarity(user_embedding, st.session_state.embeddings)[0]
                sorted_indices = cosine_similarities.argsort()[::-1]
                matches = [(df.iloc[idx]['Script Name'], cosine_similarities[idx], df.iloc[idx]['Description']) for idx in sorted_indices if cosine_similarities[idx] >= threshold]
                if not matches and recommendation:
                    matches = [(df.iloc[idx]['Script Name'], cosine_similarities[idx], df.iloc[idx]['Description']) for idx in sorted_indices[:3]]
                end_time = time.time()
                return matches, end_time - start_time

            matches, time_taken = find_best_matches(st.session_state.user_description, threshold, st.session_state.recommendation)
            if not matches:
                st.write("Sorry, no matches found. Time to code!")
                coffee_gifs = [
                    "https://media.giphy.com/media/3o6gDUfmjGOPlZRave/giphy.gif",
                    "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",
                    "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExbTR0cmgzN2hpMXBsdWgxN2xpeWpvaWlnYmJ0NmhwMDhmZzc4NGIyZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3nbxypT20Ulmo/giphy.webp",
                    "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzFrc2k5YWY5dmhlNGw4aGNwMnRrOGVuczliYzdtMWJ1cTVlZ3V4cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xULW8vRQrlIPRfiEog/giphy.webp",
                    "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMWRoNW96OTgyODBvdTRqZTRtM3Z5OWVwa2sxdm16Y29sb2dyNjh2MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WsNbxuFkLi3IuGI9NU/giphy.webp",
                    "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExeWg3NTlocWFjYjJhcDViM3A3Z2k4N3Z2ZWtyaW1hNXZlemluc3lmMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kspVl6FzbdblOMKRmM/giphy.webp",
                    "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExeWg3NTlocWFjYjJhcDViM3A3Z2k4N3Z2ZWtyaW1hNXZlemluc3lmMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/gEKz4VLX7fQlsl8SFE/giphy.webp",
                    ]
                st.image(random.choice(coffee_gifs))
            else:
                best_match = matches[0]
                st.write("### Best Match Found:")
                best_match_df = pd.DataFrame([best_match], columns=["Script Name", "Similarity Score", "Description"])
                st.dataframe(best_match_df.drop(columns=["Description"]))
                st.write(f"**Description:** {best_match[2]}")
                
                copy_link_html = f"""
                <a href="{df[df['Script Name'] == best_match[0]]['Path'].values[0]}" target="_blank" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">Open Script</a>
                """
                st.markdown(copy_link_html, unsafe_allow_html=True)
                
                if len(matches) > 1:
                    remaining_matches = matches[1:]
                    results_df = pd.DataFrame(remaining_matches, columns=["Script Name", "Similarity Score", "Description"])
                    st.write("### Other Matches Found:")
                    st.dataframe(results_df.drop(columns=["Description"]))
                    for index, row in results_df.iterrows():
                        with st.expander(f"Description for {row['Script Name']}"):
                            st.write(row['Description'])
                            open_script_html = f"""
                            <a href="{df[df['Script Name'] == row['Script Name']]['Path'].values[0]}" target="_blank" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">Open Script</a>
                            """
                            st.markdown(open_script_html, unsafe_allow_html=True)
                st.write(f"**Time taken:** {time_taken:.4f} seconds")
    else:
        st.write("Please enter a description.")
