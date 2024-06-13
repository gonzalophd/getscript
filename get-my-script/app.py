import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

# Load the CSV file
url = 'https://raw.githubusercontent.com/gonzalophd/getscript/main/get-my-script/script_DB.csv'
def load_original_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load data from GitHub.")
        return None

df = load_original_data(url)

# Check if the 'Description' column exists
if df is not None and 'Description' not in df.columns:
    st.error("The 'Description' column is missing in the CSV file. Please check the file.")
else:
    # Load a pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def find_best_matches(description, threshold=0.3, recommendation=False):
        start_time = time.time()

        # Combine the user input description with the existing descriptions
        descriptions = df['Description'].tolist()
        descriptions.append(description)

        # Generate embeddings for the descriptions
        embeddings = model.encode(descriptions)

        # Calculate cosine similarities
        cosine_similarities = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]

        # Sort indices based on similarity scores in descending order
        sorted_indices = cosine_similarities.argsort()[::-1]

        # Filter out matches below the threshold
        matches = []
        for idx in sorted_indices:
            if cosine_similarities[idx] >= threshold:
                matches.append((df.iloc[idx]['Script Name'], cosine_similarities[idx]))

        # If no matches are found and recommendation is True, provide top 2 or 3 closest matches regardless of threshold
        if not matches and recommendation:
            for idx in sorted_indices[:3]:
                matches.append((df.iloc[idx]['Script Name'], cosine_similarities[idx]))

        end_time = time.time()
        time_taken = end_time - start_time

        return matches, time_taken

    # Streamlit app
    st.title("Get my Script!")
    st.write("This app helps you find the best match for your company based on the description you provide.")

    user_description = st.text_area("Enter a description:", "")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3)
    recommendation = st.checkbox("Provide recommendations if no matches found")

    if st.button("Find Matches"):
        if user_description:
            matches, time_taken = find_best_matches(user_description, threshold, recommendation)

            if not matches:
                st.write("Sorry, we couldn't find any match.")
            else:
                # Display matches in a table
                results_df = pd.DataFrame(matches, columns=["Script Name", "Similarity Score"])
                st.write("### Best match companies:")
                st.dataframe(results_df)

                st.write(f"**Time taken to find the best matches:** {time_taken:.4f} seconds")
        else:
            st.write("Please enter a description.")
