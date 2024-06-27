import streamlit as st
import pandas as pd
import requests
from io import StringIO

# Function to navigate to a different page
def navigate_to(page):
    st.query_params(page=page)

# Load the CSV file
url = 'https://raw.githubusercontent.com/gonzalophd/getscript/main/get-my-script/script_DB_eng.csv'

@st.cache_data(ttl=86400)  # Cache for one day
def load_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to download the database from Github.")
        return None

df = load_data(url)

# Streamlit app interface
st.title("Explore the Script Database")

# Add the "Back to Main" button
if st.button("Back to Main"):
    navigate_to("main")

# Filtering options
script_name_filter = st.text_input("Script Name contains:")
description_filter = st.text_input("Description contains:")

# Apply filters
filtered_df = df.copy()

if script_name_filter:
    filtered_df = filtered_df[filtered_df['Script Name'].str.contains(script_name_filter, case=False, na=False)]

if description_filter:
    filtered_df = filtered_df[filtered_df['Description'].str.contains(description_filter, case=False, na=False)]

# Display filtered data
st.write("### Filtered Scripts")
st.dataframe(filtered_df)

# Additional details
if not filtered_df.empty:
    for index, row in filtered_df.iterrows():
        with st.expander(f"Description for {row['Script Name']}"):
            st.write(row['Description'])
