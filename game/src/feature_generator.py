import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
from game.flaskapp_andrius.api import preprocesser
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler


def clean_recipe_table(recipe_table_csv, updated_images_csv):
    """
    Takes original recipe table, updates images using new csv, then cleans data
    Input format
    clean_recipe_table('../data/recipe_table.csv', '../data/recipe_table_updated_UAT.csv')
    """
    # Read and prepare CSV
    df_old = pd.read_csv(recipe_table_csv, sep=";")
    df_new = pd.read_csv(updated_images_csv, sep=",").reset_index()
    df_new.columns = df_new.iloc[0].tolist()
    df_new = df_new.drop(0)
    df_new["id"] = df_new["id"].astype(int)

    # Make dict with IDs and URLs
    old_images = df_old.set_index("id").to_dict()["image_url"]
    new_images = df_new.set_index("id").to_dict()["image_url"]

    # Match new image links with old IDs
    new_dict = {k: new_images.get(k, v) for k, v in old_images.items()}

    # Make new df with updated images
    updated_images = pd.DataFrame.from_dict(
        new_dict, orient="index", columns=["image_url"]
    )

    # Add image resizing
    updated_images = updated_images.reset_index()
    updated_images.columns = ["id", "image_url"]
    updated_images["image_url"] = (
        updated_images["image_url"] + "?auto=format&fit=crop&w=320&h=218"
    )

    # Update old recipe table with new images
    df_updated = df_old.copy()
    df_updated["image_url"] = updated_images["image_url"]

    # Remove links broken in old and new table for certain recipes
    broken_links = [741, 808, 824, 880, 889, 938, 945]
    for i in broken_links:
        df_updated = df_updated.drop(df_updated.loc[df_updated["id"] == i].index)

    # Drop recipes without a foodgroup entry
    df_updated = df_updated.dropna(subset=["food_group"])

    # Broken links in new table that were working fine in old
    revert_links = [320, 335, 349, 495, 647, 764]
    for elem in revert_links:
        reverted_image = df_old.at[df_old[df_old["id"] == elem].index[0], "image_url"]
        df_updated.at[
            df_updated[df_updated["id"] == elem].index[0], "image_url"
        ] = reverted_image

    # Reassigning ginger salmon to fish instead of chicken
    gin_salm = "Ginger salmon with buckwheat noodles & peanuts"
    df_updated.at[
        df_updated[df_updated["title"] == gin_salm].index[0], "food_group"
    ] = "Fish"

    # Change 4 shellfish entries to correct food groups
    # IDs: 31, 102, 366, 407
    df_updated.at[df_updated[df_updated["id"] == 31].index[0], "food_group"] = "Beef"
    df_updated.at[
        df_updated[df_updated["id"] == 102].index[0], "food_group"
    ] = "Chicken"
    df_updated.at[
        df_updated[df_updated["id"] == 366].index[0], "food_group"
    ] = "Chicken"
    df_updated.at[df_updated[df_updated["id"] == 407].index[0], "food_group"] = "Fish"

    # Upload to data folder
    df_updated.to_csv("../data/recipe_table_new.csv", sep=";", index=False)

    print("---New recipe table generated as 'recipe_table_new.csv'---")


def get_embedding(recipe_dict):
    """
    Takes a recipe dictionary and calculates recipe vector.
    Take mean of recipe vector to give embedding, in line with original original get_delta12 function
    """
    list_of_embeddings = []
    for recipe in recipe_dict:
        recipe_embedding = np.mean(preprocesser.recipe2vec(recipe))
        list_of_embeddings.append(recipe_embedding)
    return list_of_embeddings


def get_vector(recipe):
    """
    Takes a single recipe and returns 45-D recipe vector
    """
    return preprocesser.recipe2vec(recipe)


def get_vector_list(recipe_dict):
    """
    Takes recipe list and returns list of recipe vectors
    """
    list_of_vectors = []
    for recipe in recipe_dict:
        recipe_vector = preprocesser.recipe2vec(recipe)
        list_of_vectors.append(recipe_vector)
    return list_of_vectors


def get_euc_dist_from_origin(recipe_array):
    """
    Takes single recipe array and calculates magnitude from the origin
    """
    recipe_array = recipe_array.reshape(1, -1)
    dist = euclidean_distances(
        recipe_array,
        [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ],
    )
    return dist


def get_top_twenty_words(df, n_comp):
    """
    Takes dataframe of text (title, description, key ingredient) and uses truncated SVD to reduce down to n components
    """
    vectorizer = TfidfVectorizer(stop_words="english", strip_accents="unicode")
    vec = vectorizer.fit_transform(df)
    tfidf = pd.DataFrame(vec.todense(), columns=vectorizer.get_feature_names())
    svd = TruncatedSVD(n_components=n_comp, n_iter=7, random_state=42)
    svd_ft = svd.fit_transform(tfidf)
    df_comp = pd.DataFrame(svd.components_, columns=tfidf.columns)
    df_comp = df_comp.T
    return np.mean(df_comp.abs(), axis=1).sort_values(ascending=False).head(10)
