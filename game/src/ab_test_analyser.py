from game.flaskapp_andrius.api import preprocesser
from game.utils import run_query
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import warnings
from game.src.delta12_calculator import (
    get_delta12_api,
    get_recipe_id_api,
    get_recipe_api,
    get_delta12,
)
from game.src.ab_test_preprocessor import (
    calc_new_delta,
    get_last_order_hist,
    get_all_orders,
    get_order_hist_from_observed,
)
from game.src.feature_generator import get_embedding, get_vector_list


def analyse_abtest(existing_order_csv, observed_order_csv, game_number):
    """
    GAME 1 AND GAME 2 ONLY
    Takes existing orders for customers who responded to game. Calculates their historical delta12 and magnitude values
    Takes customers' observed orders. Calculates their observed delta12 and magnitude values
    Then for each metric, takes the observed value and subtracts the historical value, to give a difference in metric for each customer
    Exports result to csv, for delta12 and magnitude metrics

    Example input
    from ab_test_analyser import analyse_abtest
    analyse_abtest('../data/order_history_g1.csv','../data/tableGame1.csv', 'game1')
    analyse_abtest('../data/order_history_g2.csv','../data/database2.csv', 'game2')
    """
    # Ignore deprecation warning
    warnings.warn("deprecated", DeprecationWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Import data
        order_history = pd.read_csv(existing_order_csv, sep=",")
        g_observed = pd.read_csv(observed_order_csv)

        # PREPROCESSING
        # Columns with relevant order history data
        recipe_cols = g_observed.columns[2:22]

        # Preprocess order history table
        order_history = order_history.reset_index()
        order_history.columns = order_history.iloc[0].tolist()
        order_history = order_history.drop(0)
        order_history = order_history.dropna(subset=["id"])
        order_history["id"] = order_history["id"].astype(int)
        order_history["delivery_id"] = order_history["delivery_id"].astype(int)
        order_history["customer_id"] = order_history["customer_id"].astype(int)

        # Get updated recipe info and generate recipe vectors
        # The new recipe needs to be used because recent historical orders contain recipes
        # from this table
        df_updated_recipes = pd.read_csv(
            "../data/recipe_table_updated_UAT.csv", sep=","
        )
        df_updated_recipes = df_updated_recipes.reset_index()
        df_updated_recipes.columns = df_updated_recipes.iloc[0].tolist()
        df_updated_recipes = df_updated_recipes.drop(0)
        df_updated_recipes["id"] = df_updated_recipes["id"].astype(int)
        df_updated_recipes = df_updated_recipes.dropna(subset=["key_ingredient"])
        updated_recipe_dict = df_updated_recipes.to_dict(orient="records")
        list_of_updated_vectors = get_vector_list(updated_recipe_dict)
        df_updated_recipes["vector"] = list_of_updated_vectors
        print("- - - - - Generated recipe vectors - - - - -")

        # Preprocessing for observed data from AB testing
        g_observed = g_observed.dropna(subset=["customer_id"])
        g_observed["customer_id"] = g_observed["customer_id"].astype(int)
        g_observed[recipe_cols] = g_observed[recipe_cols].replace(0, np.nan)

        # Generate recipe vectors from original recipe table
        # This recipe table was used to develop the game
        # Future iterations should use one recipe table for everything
        df_recipes = pd.read_csv("../data/recipe_table.csv", sep=";")
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
            ]
        ]
        recipe_dict = df_clean.to_dict(orient="records")
        list_of_vectors = get_vector_list(recipe_dict)
        df_clean["vector"] = list_of_vectors
        df_vectors = pd.DataFrame(df_clean[["id", "vector"]])

        # CALCULATIONS ON HISTORICAL AND OBSERVED DATA
        # Get historical delta12 for game-playing customers
        print("\n- - - - - Now calculating historical delta12 - - - - -")
        g_customers = order_history["customer_id"].unique()
        g_historical = pd.DataFrame(g_customers, columns=["customer_id"])
        g_historical["historical_delta12"] = g_historical["customer_id"].apply(
            lambda x: get_delta12(get_last_order_hist(order_history, x))
        )
        print("- - - - - Calculated historical delta12 - - - - -")

        # Get historical pairwise magnitudes for game-playing customers
        print("\n- - - - - Now calculating historical magnitude - - - - -")
        g_historical["historical_magnitude"] = g_historical["customer_id"].apply(
            lambda x: calc_new_delta(
                get_all_orders(order_history, x), df_updated_recipes
            )
        )
        print("- - - - - Calculated historical magnitude - - - - -")

        # Sort by customer ID for merge
        g_historical = g_historical.sort_values("customer_id")

        # Calculate observed delta12
        # get_delta12 only requires recipe order history to generate value
        print("\n- - - - - Now calculating observed delta12 - - - - -")
        g_observed["observed_delta12"] = g_observed["customer_id"].apply(
            lambda x: get_delta12(
                get_order_hist_from_observed(g_observed, x, recipe_cols)
            )
        )
        print("- - - - - Calculated observed delta12 - - - - -")

        # Get observed magnitudes (this takes some time to run)
        print("\n- - - - - Now calculating observed magnitude - - - - -")
        g_observed["observed_magnitude"] = 0
        for i in range(len(g_observed)):
            g_observed["observed_magnitude"].iloc[i] = calc_new_delta(
                g_observed[recipe_cols].iloc[i].dropna().tolist(), df_vectors
            )
        print("- - - - - Calculated observed magnitude - - - - -")

        # Extract and sort
        g_observed_clean = g_observed[
            ["customer_id", "observed_delta12", "observed_magnitude"]
        ].sort_values("customer_id")

        # COMBINATION OF HISTORICAL AND OBSERVED
        # Merge historical and observed data
        g_merged = pd.merge(g_historical, g_observed_clean, on="customer_id")

        # Calculate difference between observed and historical values
        g_merged["diff_delta12"] = (
            g_merged["observed_delta12"] - g_merged["historical_delta12"]
        )
        g_merged["diff_magnitude"] = (
            g_merged["observed_magnitude"] - g_merged["historical_magnitude"]
        )

        # Show statistics
        print("\nStatistics for difference in observed and historical delta12")
        print(g_merged["diff_delta12"].describe())
        print("\nStatistics for difference in observed and historical magnitude")
        print(g_merged["diff_magnitude"].describe())

        # Export output
        g_merged[["customer_id", "diff_delta12", "diff_magnitude"]].to_csv(
            "../data/" + game_number + "_results.csv", index=False
        )
        print(
            "\n- - - - - Analysis generated as '"
            + game_number
            + "_results.csv' - - - - -"
        )


def analyse_abtest_game3(existing_order_csv, observed_order_csv, game_number):
    """
    GAME 3 ONLY
    Game 3 has a different observed_order_csv structure
    Game 3 had instances where customers chose broken links. These are removed in preprocessing
    Game 3 had 1 instance where a customer answered twice

    Takes existing orders for customers who responded to game. Calculates their historical delta12 and magnitude values
    Takes customers' observed orders. Calculates their observed delta12 and magnitude values
    Then for each metric, takes the observed value and subtracts the historical value, to give a difference in metric for each customer
    Exports result to csv, for delta12 and magnitude metrics

    Example input
    from ab_test_analyser import analyse_abtest_game3
    analyse_abtest_game3('../data/order_history_g3.csv','../data/game3.csv', 'game3')
    """
    # Ignore deprecation warning
    warnings.warn("deprecated", DeprecationWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Import data
        order_history = pd.read_csv(existing_order_csv, sep=",")
        g_observed = pd.read_csv(observed_order_csv)

        # PREPROCESSING
        # Columns with relevant order history data
        recipe_cols = g_observed.columns[2:12]  # Game 3 edit

        # Remove entries where customers have chosen broken links
        for i in recipe_cols:  # Game 3 edit
            g_observed = g_observed[g_observed[i] != 3]
            g_observed = g_observed[g_observed[i] != 19]
            g_observed = g_observed[g_observed[i] != 45]
            g_observed = g_observed[g_observed[i] != 4]

        # Remove instance where someone answered twice...
        g_observed = g_observed[g_observed["record_id"] != 160]  # Game 3 edit

        # Preprocess order history table
        order_history = order_history.reset_index()
        order_history.columns = order_history.iloc[0].tolist()
        order_history = order_history.drop(0)
        order_history = order_history.dropna(subset=["id"])
        order_history["id"] = order_history["id"].astype(int)
        order_history["delivery_id"] = order_history["delivery_id"].astype(int)
        order_history["customer_id"] = order_history["customer_id"].astype(int)

        # Get updated recipe info and generate recipe vectors
        # The new recipe needs to be used because recent historical orders contain recipes
        # from this table
        df_updated_recipes = pd.read_csv(
            "../data/recipe_table_updated_UAT.csv", sep=","
        )
        df_updated_recipes = df_updated_recipes.reset_index()
        df_updated_recipes.columns = df_updated_recipes.iloc[0].tolist()
        df_updated_recipes = df_updated_recipes.drop(0)
        df_updated_recipes["id"] = df_updated_recipes["id"].astype(int)
        df_updated_recipes = df_updated_recipes.dropna(subset=["key_ingredient"])
        updated_recipe_dict = df_updated_recipes.to_dict(orient="records")
        list_of_updated_vectors = get_vector_list(updated_recipe_dict)
        df_updated_recipes["vector"] = list_of_updated_vectors
        print("- - - - - Generated recipe vectors - - - - -")

        # Preprocessing for observed data from AB testing
        g_observed = g_observed.dropna(subset=["customer_id"])
        g_observed["customer_id"] = g_observed["customer_id"].astype(int)
        g_observed[recipe_cols] = g_observed[recipe_cols].replace(0, np.nan)

        # Generate recipe vectors from original recipe table
        # This recipe table was used to develop the game
        # Future iterations should use one recipe table for everything
        df_recipes = pd.read_csv("../data/recipe_table.csv", sep=";")
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
            ]
        ]
        recipe_dict = df_clean.to_dict(orient="records")
        list_of_vectors = get_vector_list(recipe_dict)
        df_clean["vector"] = list_of_vectors
        df_vectors = pd.DataFrame(df_clean[["id", "vector"]])

        # CALCULATIONS ON HISTORICAL AND OBSERVED DATA
        # Get historical delta12 for game-playing customers
        print("\n- - - - - Now calculating historical delta12 - - - - -")
        g_customers = order_history["customer_id"].unique()
        g_historical = pd.DataFrame(g_customers, columns=["customer_id"])
        g_historical["historical_delta12"] = g_historical["customer_id"].apply(
            lambda x: get_delta12(get_last_order_hist(order_history, x))
        )
        print("- - - - - Calculated historical delta12 - - - - -")

        # Get historical pairwise magnitudes for game-playing customers
        print("\n- - - - - Now calculating historical magnitude - - - - -")
        g_historical["historical_magnitude"] = g_historical["customer_id"].apply(
            lambda x: calc_new_delta(
                get_all_orders(order_history, x), df_updated_recipes
            )
        )
        print("- - - - - Calculated historical magnitude - - - - -")

        # Sort by customer ID for merge
        g_historical = g_historical.sort_values("customer_id")

        # Calculate observed delta12
        # get_delta12 only requires recipe order history to generate value
        print("\n- - - - - Now calculating observed delta12 - - - - -")
        g_observed["observed_delta12"] = g_observed["customer_id"].apply(
            lambda x: get_delta12(
                get_order_hist_from_observed(g_observed, x, recipe_cols)
            )
        )
        print("- - - - - Calculated observed delta12 - - - - -")

        # Get observed magnitudes (this takes some time to run)
        print("\n- - - - - Now calculating observed magnitude - - - - -")
        g_observed["observed_magnitude"] = 0
        for i in range(len(g_observed)):
            g_observed["observed_magnitude"].iloc[i] = calc_new_delta(
                g_observed[recipe_cols].iloc[i].dropna().tolist(), df_vectors
            )
        print("- - - - - Calculated observed magnitude - - - - -")

        # Extract and sort
        g_observed_clean = g_observed[
            ["customer_id", "observed_delta12", "observed_magnitude"]
        ].sort_values("customer_id")

        # COMBINATION OF HISTORICAL AND OBSERVED
        # Merge historical and observed data
        g_merged = pd.merge(g_historical, g_observed_clean, on="customer_id")

        # Calculate difference between observed and historical values
        g_merged["diff_delta12"] = (
            g_merged["observed_delta12"] - g_merged["historical_delta12"]
        )
        g_merged["diff_magnitude"] = (
            g_merged["observed_magnitude"] - g_merged["historical_magnitude"]
        )

        # Show statistics
        print("\nStatistics for difference in observed and historical delta12")
        print(g_merged["diff_delta12"].describe())
        print("\nStatistics for difference in observed and historical magnitude")
        print(g_merged["diff_magnitude"].describe())

        # Export output
        g_merged[["customer_id", "diff_delta12", "diff_magnitude"]].to_csv(
            "../data/" + game_number + "_results.csv", index=False
        )
        print(
            "\n- - - - - Analysis generated as '"
            + game_number
            + "_results.csv' - - - - -"
        )
