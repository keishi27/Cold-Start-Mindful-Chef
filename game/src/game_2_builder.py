import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from game.flaskapp_andrius.api import preprocesser
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from game.src.feature_generator import get_vector_list


def get_recipe_pca(recipe_table_csv):
    """
    Using recipe text data, generates vector array for each recipe
    PCA reduces dimensionality further, to account for 95% of variance

    get_recipe_pca('../data/recipe_table.csv')
    """
    df_recipes = pd.read_csv(recipe_table_csv, sep=";")
    df_recipes["price"] = df_recipes["price_2p_pence"]
    df_clean = df_recipes[
        [
            "id",
            "food_group",
            "calories",
            "carbs",
            "fat",
            "protein",
            "cooking_time",
            "title",
            "description",
            "key_ingredient",
            "price",
            "image_url",
        ]
    ]
    recipe_dict = df_clean.to_dict(orient="records")

    # Make new dataframe that contains vector
    list_of_vectors = get_vector_list(recipe_dict)
    df_clean["vector"] = list_of_vectors
    df_vector = pd.DataFrame(df_clean[["id", "vector", "image_url"]])

    # Expand 45D vector into new dataframe
    df_vector_sep = pd.DataFrame(data=df_vector["vector"][0])
    df_vector_sep = df_vector_sep.transpose()

    # start at 1 since we have used the index 0 to make the new dataframe
    for i in range(1, len(df_recipes)):
        vec = pd.DataFrame(data=df_vector["vector"][i])
        vec = vec.transpose()
        df_vector_sep = df_vector_sep.append(vec)
    df_vector_sep = df_vector_sep.reset_index(drop=True)

    # Scale sparated vectors in preparation for PCA
    scaler = MinMaxScaler()
    df_rescaled = scaler.fit_transform(df_vector_sep)

    # Generate enough components to account for 95% of the variance
    pca_95_var = PCA(n_components=0.95)
    pca_95_var_ft = pca_95_var.fit_transform(df_rescaled)
    df_pca = pd.DataFrame(pca_95_var_ft)

    # Generate new dataframe that contains all the principal components
    df_final_pca = pd.DataFrame(df_vector["id"])
    df_final_pca = pd.concat([df_final_pca, df_pca], axis=1)
    col_list = ["id"]

    # Numbering of columns, start numbering at 1
    for i in range(1, len(df_final_pca.columns)):
        col_list.append("PC_" + str(i))

    df_final_pca.columns = col_list
    df_final_pca[["title", "food_group", "key_ingredient", "image_url"]] = df_clean[
        ["title", "food_group", "key_ingredient", "image_url"]
    ]

    # Generate CSV with all principal components
    df_final_pca.to_csv("../data/recipe_pca.csv", index=False)
    print(
        "- - - - - Generated recipe principal components as 'recipe_pca.csv'- - - - -"
    )


def get_recipe_3_pc(recipe_table_csv):
    """
    Using recipe text data, generates vector array for each recipe
    PCA reduces dimensionality further, then take top 3 principal components
    get_recipe_3_pc('../data/recipe_table.csv')
    """
    df_recipes = pd.read_csv(recipe_table_csv, sep=";")
    df_recipes["price"] = df_recipes["price_2p_pence"]
    df_clean = df_recipes[
        [
            "id",
            "food_group",
            "calories",
            "carbs",
            "fat",
            "protein",
            "cooking_time",
            "title",
            "description",
            "key_ingredient",
            "price",
            "image_url",
        ]
    ]
    recipe_dict = df_clean.to_dict(orient="records")

    # Make new dataframe that contains vector
    list_of_vectors = get_vector_list(recipe_dict)
    df_clean["vector"] = list_of_vectors
    df_vector = pd.DataFrame(df_clean[["id", "vector", "image_url"]])

    # Expand 45D vector into new dataframe
    df_vector_sep = pd.DataFrame(data=df_vector["vector"][0])
    df_vector_sep = df_vector_sep.transpose()

    # start at 1 since we have used the index 0 to make the new dataframe
    for i in range(1, len(df_recipes)):
        vec = pd.DataFrame(data=df_vector["vector"][i])
        vec = vec.transpose()
        df_vector_sep = df_vector_sep.append(vec)
    df_vector_sep = df_vector_sep.reset_index(drop=True)

    # Scale sparated vectors in preparation for PCA
    scaler = MinMaxScaler()
    df_rescaled = scaler.fit_transform(df_vector_sep)

    # Generate enough components to account for 95% of the variance
    pca_95_var = PCA(n_components=0.95)
    pca_95_var_ft = pca_95_var.fit_transform(df_rescaled)
    df_pca = pd.DataFrame(pca_95_var_ft)

    # Generate new dataframe that contains all the principal components
    df_final_pca = pd.DataFrame(df_vector["id"])
    df_final_pca = pd.concat([df_final_pca, df_pca], axis=1)
    col_list = ["id"]

    # Numbering of columns, start numbering at 1
    for i in range(1, len(df_final_pca.columns)):
        col_list.append("PC_" + str(i))

    df_final_pca.columns = col_list
    df_final_pca[["title", "food_group", "key_ingredient", "image_url"]] = df_clean[
        ["title", "food_group", "key_ingredient", "image_url"]
    ]

    # Get top 3 principal components
    df_3_pc = df_final_pca[
        [
            "id",
            "title",
            "food_group",
            "key_ingredient",
            "image_url",
            "PC_1",
            "PC_2",
            "PC_3",
        ]
    ].copy()

    # Generate csv
    df_3_pc.to_csv("../data/recipe_3_pc.csv", index=False)
    print(
        "- - - - - Generated recipe top 3 principal components as 'recipe_3_pc.csv'- - - - -"
    )


def get_scaled_pc_by_fg(df, food_group):
    """
    Takes dataframe and food group, calculates minmax scaled PC for PC1, PC2, PC3
    Returns csv where each principal component is scaled between 0 and 1
    Input csv must be generated from get_recipe_3_pc,the recipe table with top 3 principal components
    e.g. recipe_3_pc.csv

    These csvs are used by ../game_two/app/app.py to generate images for Game 2
    """
    df_pca = df.copy()
    df_pca = df_pca[df_pca["food_group"] == food_group]
    pc_scaler = MinMaxScaler()
    df_pca_scale = pd.DataFrame(
        pc_scaler.fit_transform(df_pca["PC_1"].values.reshape((-1, 1))),
        columns=["FG_Scaled_PC_1"],
    )
    df_pca.reset_index(drop=True, inplace=True)
    df_pca = pd.concat([df_pca, df_pca_scale], axis=1)

    df_pca_scale = pd.DataFrame(
        pc_scaler.fit_transform(df_pca["PC_2"].values.reshape((-1, 1))),
        columns=["FG_Scaled_PC_2"],
    )
    df_pca.reset_index(drop=True, inplace=True)
    df_pca = pd.concat([df_pca, df_pca_scale], axis=1)

    df_pca_scale = pd.DataFrame(
        pc_scaler.fit_transform(df_pca["PC_3"].values.reshape((-1, 1))),
        columns=["FG_Scaled_PC_3"],
    )
    df_pca.reset_index(drop=True, inplace=True)
    df_pca = pd.concat([df_pca, df_pca_scale], axis=1)

    df_pca.to_csv("../data/" + str(food_group) + "_FG_Scaled_PC.csv", index=False)
    print("CSV generated for " + str(food_group))
