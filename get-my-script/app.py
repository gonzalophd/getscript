import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time


# Load the CSV file
df = pd.read_csv('C:\Users\Gonzalo\Documents\06_For_the_GHub\getscript\get-my-script\DB_companies.csv')

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
            matches.append((df.iloc[idx]['Company Name'], df.iloc[idx]['Category'], df.iloc[idx]['Subcategory'], cosine_similarities[idx]))
    
    # Extract categories and subcategories for the top matches
    categories = list(set((df.iloc[idx]['Category'], df.iloc[idx]['Subcategory']) for idx in sorted_indices if cosine_similarities[idx] >= threshold))
    
    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time
    
    # If no matches are found and recommendation is True, provide top 2 or 3 closest matches regardless of threshold
    if not matches and recommendation:
        for idx in sorted_indices[:3]:
            matches.append((df.iloc[idx]['Company Name'], df.iloc[idx]['Category'], df.iloc[idx]['Subcategory'], cosine_similarities[idx]))
        categories = list(set((df.iloc[idx]['Category'], df.iloc[idx]['Subcategory']) for idx in sorted_indices[:3]))
    
    return matches, categories, time_taken

# Streamlit app
st.title("Company Description Matcher")

user_description = st.text_area("Enter a description:", "")
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3)
recommendation = st.checkbox("Provide recommendations if no matches found")

if st.button("Find Matches"):
    if user_description:
        matches, categories, time_taken = find_best_matches(user_description, threshold, recommendation)
        
        if not matches:
            st.write("Sorry, we couldn't find any match.")
        else:
            st.write("### Best match companies in order:")
            for company, category, subcategory, score in matches:
                st.write(f"**Company Name:** {company}, **Category:** {category}, **Subcategory:** {subcategory}, **Similarity Score:** {score:.4f}")
            
            st.write("### Categories and Subcategories where the description could fit:")
            for category, subcategory in categories:
                st.write(f"**Category:** {category}, **Subcategory:** {subcategory}")
            
            st.write(f"**Time taken to find the best matches:** {time_taken:.4f} seconds")
    else:
        st.write("Please enter a description.")
