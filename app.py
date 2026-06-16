import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process

# Load the pre-trained TF-IDF Vectorizer and df_percent
# Make sure these .pkl files are in the same directory as app.py or provide the full path
try:
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    with open('df_percent.pkl', 'rb') as file:
        df_percent = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'tfidf_vectorizer.pkl' or 'df_percent.pkl' not found.")
    st.write("Please ensure these files are in the same directory as the app.py.")
    st.stop()

# Re-create the tfidf_matrix using the loaded vectorizer
tfidf_matrix = tfidf.transform(df_percent['reviews_list'])

# Create indices Series
indices = pd.Series(df_percent.index, index=df_percent['name']).drop_duplicates()

# Calculate cosine similarities
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(name, cosine_similarities=cosine_similarities):
    recommend_restaurant = []

    # Standardize input name to title case for better matching
    search_name = name.title()

    # First, try to find an exact match (case-insensitive)
    # Create a lowercased version of the indices index for lookup
    lower_case_indices_index = indices.index.str.lower()
    if search_name.lower() in lower_case_indices_index:
        # Get the actual name from the original index
        matched_name = indices.index[lower_case_indices_index == search_name.lower()][0]
        idx = indices.loc[matched_name]
    else:
        # If no exact case-insensitive match, perform fuzzy search
        fuzzy_match = process.extractOne(search_name, indices.index.tolist())

        # Define a minimum score for a fuzzy match to be considered valid
        MIN_FUZZY_SCORE = 85 # Adjust this threshold as needed

        if fuzzy_match and fuzzy_match[1] >= MIN_FUZZY_SCORE:
            matched_name = fuzzy_match[0]
            st.write(f"Using fuzzy matched restaurant: '{matched_name}' (Score: {fuzzy_match[1]})")
            idx = indices.loc[matched_name]
        else:
            st.write(f"Restaurant '{name}' not found and no close fuzzy match (score < {MIN_FUZZY_SCORE}) in the dataset.")
            return pd.DataFrame(columns=['name', 'cuisines', 'Mean Rating', 'cost'])

    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    top30_indexes = list(score_series.iloc[0:31].index)

    for each_idx in top30_indexes:
        recommend_restaurant.append(df_percent.iloc[each_idx]['name'])

    # Creating a list to collect data for similar restaurants
    all_restaurant_data = []

    # Collect data for the top 30 similar restaurants with some of their columns
    for each_name in recommend_restaurant:
        # Filter df_percent by restaurant name (from 'name' column) to get the relevant row
        restaurant_data = df_percent[['name', 'cuisines', 'Mean Rating', 'cost']][df_percent['name'] == each_name].sample(random_state=42)
        all_restaurant_data.append(restaurant_data)

    # Concatenate all collected data into a single DataFrame and reset index
    df_new = pd.concat(all_restaurant_data, ignore_index=True)

    df_new = df_new.drop_duplicates(subset=['name', 'cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)

    return df_new

# Streamlit UI
st.title('Restaurant Recommendation System')

restaurant_name = st.text_input('Enter a restaurant name:', 'Grand Village')

if st.button('Get Recommendations'):
    if restaurant_name:
        with st.spinner('Generating recommendations...'):
            recommendations = recommend(restaurant_name)
            if not recommendations.empty:
                st.write(f"### TOP 10 RESTAURANTS LIKE {restaurant_name} WITH SIMILAR REVIEWS:")
                st.dataframe(recommendations, hide_index=True) # Added hide_index=True
            else:
                st.write(f"No recommendations found for '{restaurant_name}'.")
    else:
        st.warning('Please enter a restaurant name to get recommendations.')
