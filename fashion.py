import streamlit as st
import pandas as pd
import numpy as np
import random

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('fashion_dataset.csv')

df = load_data()

# Function to get recommendations
def get_recommendations(gender, category, base_color, season, n_recommendations=5):
    # First try exact match with base color
    filtered_df = df[
        (df['Gender'] == gender) & 
        (df['Category'] == category) & 
        (df['Base_color'].str.lower() == base_color.lower()) &  # Case-insensitive color matching
        (df['Season'] == season)
    ]
    
    # If no matches found with exact base color, keep other filters strict
    if len(filtered_df) == 0:
        filtered_df = df[
            (df['Gender'] == gender) & 
            (df['Category'] == category) & 
            (df['Season'] == season)
        ]
        # Add warning about no exact color matches
        st.warning(f"No exact matches found for {base_color} color. Showing other available colors.")
    
    # If still no matches, return empty DataFrame
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    # Randomly select n_recommendations or all available items if less
    n_items = min(n_recommendations, len(filtered_df))
    selected_indices = random.sample(range(len(filtered_df)), n_items)
    
    return filtered_df.iloc[selected_indices]

# Streamlit UI
st.title('Fashion Recommendation App')

# User input form
st.subheader('Enter your preferences:')
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', sorted(df['Gender'].unique()))
    category = st.selectbox('Category', sorted(df['Category'].unique()))

with col2:
    # Normalize color names in the dataset
    base_colors = sorted(df['Base_color'].str.title().unique())
    base_color = st.selectbox('Base Color', base_colors)
    season = st.selectbox('Season', sorted(df['Season'].unique()))

if st.button('Get Recommendations', type='primary'):
    recommendations = get_recommendations(gender, category, base_color, season)
    
    if len(recommendations) > 0:
        st.subheader('Recommended Items:')
        
        for i, row in recommendations.iterrows():
            with st.container():
                st.write(f"**{row['Display_product_name']}**")
                st.write(f"Color: {row['Base_color']}")  # Display the actual color
                st.write(f"[View Product Image]({row['link_image']})")
                st.write('---')
    else:
        st.error(f"No recommendations found for {gender}'s {category} in {season} season. Try different combinations!")

# Add some information about the app
st.sidebar.title('About')
st.sidebar.info('''
    This app uses machine learning to recommend fashion items based on your preferences. 
    The recommendations are generated using filtering and randomization to provide varied suggestions.
    
    Each time you click 'Get Recommendations', you'll get a different set of suggestions from your matches!
''')