import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import numpy as np
import plotly.express as px
import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from sklearn.utils import shuffle
import faiss
import numpy as np
import spacy
spacy.cli.download("en_core_web_md")



st.title("Recipe recommendation system")

recipes = pd.read_csv('RAW_recipes.csv')
recipes = recipes.drop(['id', 'contributor_id', 'submitted', 'nutrition', 'description', 'tags'], axis=1)
recipes['ingredients'] = recipes['ingredients'].str.replace("[\[\]']", "", regex=True)
recipes['steps'] = recipes['steps'].str.replace("[\[\]']", "", regex=True)

steps = recipes['steps'].astype(str).tolist()

with open('faiss_index.pkl', 'rb') as index_file:
    index = pickle.load(index_file)

embedding_matrix = np.load('embedding_matrix.npy')

nlp = spacy.load('en_core_web_md')

def embed_text(text):
    doc = nlp(text)
    return doc.vector

def semantic_search(query, index, embedding_matrix, df, k=5):
    query_embedding = embed_text(query)

    query_embedding = np.expand_dims(query_embedding, axis=0)

    faiss.normalize_L2(query_embedding)

    _, result_indices = index.search(query_embedding, k)

    search_results = df.iloc[result_indices.flatten()]
    
    return search_results


st.title('Search for query')

query = st.text_input('Enter your recipe query:')
if query:
    result_df = semantic_search(query, index, embedding_matrix, recipes)
    st.subheader('Search Results:')
    st.table(result_df.style.set_table_styles([dict(selector='th', props=[('max-width', '100px')])]).set_properties(**{'max-height': '100px', 'font-size': '14px'}))


#######################
data = pd.read_csv("predict__igredients_steps_df.csv").dropna()

data['ingredients'] = data['ingredients'].str.replace("[\[\]']", "", regex=True)
data['steps'] = data['steps'].str.replace("[\[\]']", "", regex=True)
data['tags'] = data['tags'].str.replace("[\[\]']", "", regex=True)

pd.set_option("styler.render.max_elements", 1621452)

def get_similar_recipes(selected_recipe_cluster, data, num_recipes=5):

    similar_recipes = data[data['class'] == selected_recipe_cluster]

    similar_recipes = shuffle(similar_recipes)

    top_similar_recipes = similar_recipes.head(num_recipes)

    return top_similar_recipes

st.title("Recipe Predictor based in other recipes")
search_term = st.text_input("Search for a recipe:", "")

filtered_recipes = data[data['name'].str.contains(search_term, case=False)]

selected_recipe_name = st.selectbox("Select a recipe:", filtered_recipes['name'].tolist())

if selected_recipe_name:
    selected_recipe = data[data['name'] == selected_recipe_name]
    selected_recipe_cluster = selected_recipe['class'].values[0] 
    similar_recipes = get_similar_recipes(selected_recipe_cluster, data, num_recipes=5) 

    st.subheader("Suggested Recipes:")
    st.table(similar_recipes.drop(['tags', 'ingredients_steps', 'steps_processed'], axis=1).style.set_table_styles([dict(selector='th', props=[('max-width', '100px')])]).set_properties(**{'max-height': '100px', 'font-size': '14px'}))


filtered_data = data.drop(['ingredients_steps', 'steps_processed', 'class'], axis=1)
filtered_data = filtered_data[((filtered_data['minutes'] > 0) | (filtered_data['n_steps'] > 0)) & (filtered_data['minutes'] <= 300)]


st.header("Filtering Options")

all_tags = filtered_data['tags'].str.split(', ', expand=True).stack().unique()
selected_tags = st.multiselect('Select Tags', all_tags)

all_ingredients = filtered_data['ingredients'].str.split(', ', expand=True).stack().unique()
selected_ingredients = st.multiselect('Select Ingredients', all_ingredients)

selected_num_steps = st.slider("Select the number of steps:", min_value=min(filtered_data['n_steps']), max_value=max(filtered_data['n_steps']), value=0)

selected_num_ingredients = st.slider("Select the number of ingredients:", min_value=min(filtered_data['n_ingredients']), max_value=max(filtered_data['n_ingredients']), value=0)

max_prep_time = st.slider("Maximum preparation time (minutes):", min_value=min(filtered_data['minutes']), max_value=max(filtered_data['minutes']), value=0)

filter_button = st.button("Filter Recipes")

if filter_button:
    filtered_data_2 = filtered_data.copy()  

    tags_condition = (
            filtered_data_2['tags'].apply(lambda tags: any(tag in tags for tag in selected_tags) if tags else False)
        )

    ingredients_condition = (
        filtered_data_2['ingredients'].str.split(', ').apply(lambda x: any(ingredient in x for ingredient in selected_ingredients))
    )
    combined_condition = (tags_condition | ingredients_condition)

     
    filtered_data_2 = filtered_data_2[
        (filtered_data_2['n_steps'] == selected_num_steps) |
        (filtered_data_2['n_ingredients'] == selected_num_ingredients) |
        (filtered_data_2['minutes'] == max_prep_time)
    ]

    st.table(filtered_data_2)
