import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('fashion_dataset.csv')

df = load_data()

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Combine relevant features for recommendation
df['combined_features'] = df['Gender'] + ' ' + df['Category'] + ' ' + df['Article_type'] + ' ' + df['Base_color'] + ' ' + df['Season'] + ' ' + df['Usage']

# Create TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(user_input):
    user_tfidf = tfidf.transform([user_input])
    sim_scores = cosine_similarity(user_tfidf, tfidf_matrix)
    sim_scores = sim_scores.flatten()
    recommended_indices = sim_scores.argsort()[-5:][::-1]
    return df.iloc[recommended_indices]

# Streamlit UI
st.title('Fashion Recommendation App')

# User input form
st.subheader('Enter your preferences:')
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', df['Gender'].unique())
    category = st.selectbox('Category', df['Category'].unique())
    article_type = st.selectbox('Article Type', df['Article_type'].unique())

with col2:
    base_color = st.selectbox('Base Color', df['Base_color'].unique())
    season = st.selectbox('Season', df['Season'].unique())
    usage = st.selectbox('Usage', df['Usage'].unique())

if st.button('Get Recommendations', type='primary'):
    user_input = f"{gender} {category} {article_type} {base_color} {season} {usage}"
    recommendations = get_recommendations(user_input)
    
    st.subheader('Recommended Items:')
    
    for i, row in recommendations.iterrows():
        with st.container():
            st.write(f"**{row['Display_product_name']}**")
            st.write(f"[View Product Image]({row['link_image']})")
            st.write('---')

# Add some information about the app
st.sidebar.title('About')
st.sidebar.info('''
    This app uses machine learning to recommend fashion items based on your preferences. 
    The recommendations are generated using TF-IDF and cosine similarity on a dataset of fashion items.
''')

