# Final-project-Ironhack

##Recipe recommendation system

*[Amanda Brognoli Donini]*

*[Data Analytics Bootcamp - Ironhack - February 2024]*

## Content
- [Project Description](#project-description)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Files](#files)
- [Links](#links)

<a name="project-description"></a>

## Project Description

The struggle to find recipes that we like is real, at least for me. Based on that, I decided to create a recipe recommendation system based on:
- other recipes;
- queries;
- filters for ingredients, tags, number of ingredients, number of steps or how many minutes it takes for the recipe 

<a name="methodology"></a>

## Methodology

For the recommendation system based on other recipes: 
- Natural Language Processing (NLP);
- Bag of words;
- Kmeans clusters.

For the recommendation system based on queries:
- Semantic Search with FAISS.

For the recommendation system based on filters:
- Filtering system of the Streamlit app.

The final product was made in Streamlit, so the user can search recipes in the way that they prefer.

<a name="dataset"></a>

## Dataset

The dataset used for the project is a recipe dataset from Food.com and can be find on kaggle (RAW_recipes.csv). You can find the link for the dataset at the links section.

<a name="files"></a>

## Files
- develompment.ipynb: code used for the cleaning of the dataset and the development of the NLP and the clustering 
- semanthic_search_colab.ipynb: code used for the semantic search with FAISS
- main.py: code used for the development of the streamlit app
- requirements.txt: here you can find the libraries used in the project and their versions

<a name="links"></a>

## Links

[Kaggle dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

