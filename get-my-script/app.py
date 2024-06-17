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
        st.error("Falló la descarga de la base de datos desde Github.")
        return None

df = load_original_data(url)

# Check if the 'Descripcion' column exists
if df is not None and 'Descripcion' not in df.columns:
    st.error("La columna 'Descripcion' falta en el archivo script_DB. Por favor, checa este archivo.")
else:
    # Load a pre-trained Spanish-only SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    # Precompute embeddings for existing descriptions
    df['embeddings'] = df['Descripcion'].apply(lambda x: model.encode(x))

    def find_best_matches(description, threshold=0.3, recommendation=False):
        start_time = time.time()

        # Generate embedding for the user input description
        user_embedding = model.encode(description)

        # Calculate cosine similarities
        cosine_similarities = cosine_similarity([user_embedding], list(df['embeddings']))[0]

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
    st.write("Esta app te ayuda a encontrar el script que más se parezca a lo que necesitas.")

    user_description = st.text_area("Ingresa una descripción:", "")
    threshold = st.slider("Factor de similitud", 0.0, 1.0, 0.3)
    recommendation = st.checkbox("Entregame recomendaciones si no hay matches exactos.")

    if st.button("Get My Script!"):
        if user_description:
            matches, time_taken = find_best_matches(user_description, threshold, recommendation)

            if not matches:
                st.write("Lo siento, no encontré nada.")
            else:
                # Display matches in a table
                results_df = pd.DataFrame(matches, columns=["Script Name", "Factor de similitud"])
                st.write("### Lo mejor que encontré:")
                st.dataframe(results_df)

                st.write(f"**Me tomó:** {time_taken:.4f} segundos")
        else:
            st.write("Por favor escribe una descripción.")
