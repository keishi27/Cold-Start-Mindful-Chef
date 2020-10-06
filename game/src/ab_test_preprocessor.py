import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def calc_new_delta(recipe_ids, df_vectors):
    """
    Takes order list of recipe IDs. Calculates pairwise magnitudes between all recipes. Returns mean of magnitudes
    """
    # recipe_ids = list(order_dict.values())
    list_of_magnitudes = []
    if len(recipe_ids) < 2:
        list_of_magnitudes.append(0)
    else:
        # Take only the last 10 recipes, otherwise takes too long
        for i in recipe_ids[-10:]:
            for j in recipe_ids[-10:]:
                if i != j:
                    magnitude = euclidean_distances(
                        np.array(
                            df_vectors[df_vectors["id"] == i]["vector"].values.tolist()
                        ).reshape((1, -1)),
                        np.array(
                            df_vectors[df_vectors["id"] == j]["vector"].values.tolist()
                        ).reshape((1, -1)),
                    )
                    list_of_magnitudes.append(magnitude)
    return np.mean(list_of_magnitudes)


def get_last_order_hist(df, customer_id):
    """
    Get last 2 orders for customer ID. Dataframe: order_history_g2
    """
    list_of_deliveries = df[df["customer_id"] == customer_id]["delivery_id"].unique()
    last_order = []
    last_order2 = []
    # If only one order, take the same order twice
    if len(list_of_deliveries) < 2:
        for i in range(len(df[df["delivery_id"] == list_of_deliveries[-1]])):
            last_order.append(
                df[df["delivery_id"] == list_of_deliveries[-1]].iloc[i].to_dict()
            )
            last_order2.append(
                df[df["delivery_id"] == list_of_deliveries[-1]].iloc[i].to_dict()
            )
    # If 2 or more orders, take last order and order before last
    else:
        for i in range(len(df[df["delivery_id"] == list_of_deliveries[-1]])):
            last_order.append(
                df[df["delivery_id"] == list_of_deliveries[-1]].iloc[i].to_dict()
            )
        for i in range(len(df[df["delivery_id"] == list_of_deliveries[-2]])):
            last_order2.append(
                df[df["delivery_id"] == list_of_deliveries[-2]].iloc[i].to_dict()
            )
    keys = ["lastorder", "lastorder2"]
    values = [last_order, last_order2]

    return dict(zip(keys, values))


def get_all_orders(df, customer_id):
    """
    Get all orders for customer ID
    """
    return df[df["customer_id"] == customer_id]["id"].tolist()


def get_order_hist_from_observed(g_observed, customer_id, recipe_cols):
    """
    Takes observed dataframe, and extracts last_order and last_order2, to be submitted to get_delta12
    """
    old_recipes = pd.read_csv("../data/recipe_table.csv", sep=";")
    entries_per_order = int(len(recipe_cols) / 2)
    last_order_cols = recipe_cols[:entries_per_order]
    last_order2_cols = recipe_cols[entries_per_order:]

    last_order = (
        g_observed[g_observed["customer_id"] == customer_id][last_order_cols]
        .iloc[0]
        .dropna()
        .astype(int)
        .tolist()
    )
    last_order2 = (
        g_observed[g_observed["customer_id"] == customer_id][last_order2_cols]
        .iloc[0]
        .dropna()
        .astype(int)
        .tolist()
    )

    last_order_dict = []
    last_order2_dict = []
    for i in range(len(last_order)):
        last_order_dict.append(
            old_recipes[old_recipes["id"] == last_order[i]].iloc[0].to_dict()
        )

    for i in range(len(last_order2)):
        last_order2_dict.append(
            old_recipes[old_recipes["id"] == last_order2[i]].iloc[0].to_dict()
        )

    keys = ["lastorder", "lastorder2"]
    values = [last_order_dict, last_order2_dict]

    return dict(zip(keys, values))


def get_order_hist_from_observed_debug(g_observed, customer_id, recipe_cols):
    """
    Debugging module for get order hist function
    """
    try:
        old_recipes = pd.read_csv("../data/recipe_table.csv", sep=";")
        entries_per_order = int(len(recipe_cols) / 2)
        last_order_cols = recipe_cols[:entries_per_order]
        last_order2_cols = recipe_cols[entries_per_order:]

        last_order = (
            g_observed[g_observed["customer_id"] == customer_id][last_order_cols]
            .iloc[0]
            .dropna()
            .astype(int)
            .tolist()
        )
        last_order2 = (
            g_observed[g_observed["customer_id"] == customer_id][last_order2_cols]
            .iloc[0]
            .dropna()
            .astype(int)
            .tolist()
        )

        last_order_dict = []
        last_order2_dict = []
        for i in range(len(last_order)):
            last_order_dict.append(
                old_recipes[old_recipes["id"] == last_order[i]].iloc[0].to_dict()
            )

        for i in range(len(last_order2)):
            last_order2_dict.append(
                old_recipes[old_recipes["id"] == last_order2[i]].iloc[0].to_dict()
            )

        keys = ["lastorder", "lastorder2"]
        values = [last_order_dict, last_order2_dict]

        return dict(zip(keys, values))
    except:
        print("Error with " + str(customer_id))
