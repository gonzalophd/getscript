import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests

# Load data from GitHub URL
url = 'https://raw.githubusercontent.com/gonzalophd/getscript/main/get-my-script/script_DB.csv'

@st.cache_data
def load_data(url):
    response = requests.get(url)
    with open('script_DB.csv', 'wb') as file:
        file.write(response.content)
    data = pd.read_csv('script_DB.csv')
    return data

data = load_data(url)

# Check the columns of the dataframe
st.write("Columns in the dataset:", data.columns.tolist())

# Use the correct column names
description_col = 'Description'
name_col = 'Script Name'

if description_col not in data.columns or name_col not in data.columns:
    st.error(f"Expected columns '{description_col}' and '{name_col}' not found in the dataset.")
else:
    # Initialize Sentence Transformer model
    @st.cache_data
    def initialize_model():
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        corpus_embeddings = model.encode(data[description_col].tolist(), convert_to_tensor=True)
        return model, corpus_embeddings

    model, corpus_embeddings = initialize_model()

    # Streamlit app
    st.title('Script Finder App')
    st.write("Enter a description to find the best matching scripts:")

    user_input = st.text_area("Description")

    if st.button('Find Scripts'):
        if user_input:
            query_embedding = model.encode(user_input, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            data['similarity'] = cosine_scores.cpu().numpy()
            results = data.sort_values(by='similarity', ascending=False).head(5)

            st.write("Top Matching Scripts:")
            for index, row in results.iterrows():
                st.write(f"**Script Name:** {row[name_col]}")
                st.write(f"**Description:** {row[description_col]}")
                st.write(f"**Similarity Score:** {row['similarity']:.4f}")
                st.write("---")
        else:
            st.write("Please enter a description to find matching scripts.")
