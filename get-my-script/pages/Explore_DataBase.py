import streamlit as st
import pandas as pd
import requests
import io
from io import StringIO

# Load the CSV file
url = 'https://raw.githubusercontent.com/gonzalophd/getscript/main/get-my-script/script_DB_eng.feather'

def load_original_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Read the response content as bytes
        df = pd.read_feather(io.BytesIO(response.content))
        # Convert all relevant columns to lowercase
        df['Description'] = df['Description'].str.lower()
        df['Category/Software'] = df['Category/Software'].str.lower()
        df['Keywords'] = df['Keywords'].str.lower()
        df['Combined'] = df['Description'] + ' ' + df['Category/Software'] + ' ' + df['Keywords']
        return df
    else:
        print("Failed to download the database from Github.")
        return None
    
df = load_original_data(url)

# Streamlit app interface
st.title("Explore the Database")
st.write("Explore the database by filtering by Combined Description.")

# Text input for filtering by combined description
combined_filter = st.text_input("Filter by comma separated keywords:")

# Apply filters
filtered_df = df.copy()  # Use a copy of the dataframe to apply filters

if combined_filter:
    filtered_df = filtered_df[filtered_df['Combined'].str.contains(combined_filter, case=False, na=False)]

# Display filtered data
if not filtered_df.empty:
    st.write("### Filtered Scripts")
    for index, row in filtered_df.iterrows():
        st.markdown(f"<h2 style='font-size: 24px;'>{row['Script Name']}</h2>", unsafe_allow_html=True)
        st.write(f"{row['Combined']}")
        button_html = f"""
        <a href="{row['Path']}" target="_blank" style="background-color: #4CAF50; color: white; padding: 5px 10px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">Open Script</a>
        """
        st.markdown(button_html, unsafe_allow_html=True)
        st.markdown("""<hr style="border: none; border-top: 1px solid #4CAF50;"/>""", unsafe_allow_html=True)
else:
    st.write("No scripts found with the given filters.")
