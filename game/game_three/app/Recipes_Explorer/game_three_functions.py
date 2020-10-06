#!/usr/bin/env python
# coding: utf-8

# In[43]:


# Functions for game 3
# To run locally, uncomment the parameters below
import pandas as pd
import numpy as np
import random
from loguru import logger

# parameters that are inserted by developer
# options = 20
# n_of_clusters = 4
# n_recipes_in_order = 5
# n_recipes_per_cluster = options/n_of_clusters

# parameters retrieved by data frame
# n_recipes = 686
# n_choices = random.randint(5,options)
# ids_chosen = [random.randint(1,n_recipes) for i in range(n_choices) ]
# clusters = [random.randint(1,n_clusters) for i in range(n_choices) ]

# function 1
def get_equal_recipe_id_from_clusters(df, column, n_of_clusters, n_recipes_per_cluster):
    """Select equal number of recipes from each cluster.
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        data containing two columns: cluster and recipe_id
    column: pd.series
        name of column to filter, typically cluster
    n_of_clusters: int
    n_recipes_per_cluster: int
        number of recipes of recipes sampled randomly from each cluster
    
    Returns
    -------
    rdm_list_of_recipes: list
        formed of "number_of_clusters" lists, each formed by "n_recipes_per_cluster" 
        random recipe_ids inside a specific cluster 
    """
    df = df.copy()
    logger.info(df.head())
    rdm_list_of_recipes = [df[df[column] == i].sample(n=n_recipes_per_cluster)["id"].tolist() for i in range(n_of_clusters)]
    """[
        logger.info(f"Recipes in cluster {i}: {rdm_list_of_recipes[i]}\n")
        for i in range(n_of_clusters)
    ]"""

    return rdm_list_of_recipes


# function 2
# checks: sum(cluster_recipe_dict.values()) can't be greater than 5:
def get_number_of_recipes_per_cluster(df_choice, column, n_recipes_in_order):

    """Determine the number of recipes from each cluster according to customer's choice.
    Parameters
    ----------
    df_choice: pandas.core.frame.DataFrame
        data containing two columns: cluster and recipe_id and n_choices rows
    column: string
        name of column to filter, namely the cluster column
    n_recipes_in_order: int
        number of recipes that build up an order

    Returns
    -------
    cluster_recipe_dict: dictionary
        dictionary with the following parameters:
        keys are the clusters names, for example: (0, 1, 2, 3).
        values are the number of recipes for each cluster
    """
    df_choice = df_choice.copy()

    clusters = df_choice[column].tolist()
    n_choices = df_choice.shape[0]
    customer_choices = {
        cluster: df_choice[df_choice[column] == cluster].shape[0]
        for cluster in set(clusters)
    }
    percentages = {
        cluster: customer_choices[cluster] / n_choices for cluster in set(clusters)
    }

    if sum(percentages.values()) > 1:
        print(percentages.values())
        raise Exception("Probabilities must add up to 1.")

    n_recipes_to_pick = [
        round(percentage * n_recipes_in_order) for percentage in percentages.values()
    ]
    cluster_recipe_dict = dict(zip(set(clusters), n_recipes_to_pick))

    if sum(cluster_recipe_dict.values()) > n_recipes_in_order:
        min_key = min(cluster_recipe_dict, key=cluster_recipe_dict.get)
        # check what to do when three values are equal to 1
        # d = {'a':3, 'b':1, 'c':1, 'd':1}

        if cluster_recipe_dict[min_key] != 0:
            del cluster_recipe_dict[min_key]
        else:
            cluster_recipe_dict[min_key] == 0
            del cluster_recipe_dict[min_key]
            min_key = min(cluster_recipe_dict, key=cluster_recipe_dict.get)
            del cluster_recipe_dict[min_key]

    if sum(cluster_recipe_dict.values()) < n_recipes_in_order:
        max_key = max(cluster_recipe_dict, key=cluster_recipe_dict.get)
        cluster_recipe_dict[max_key] = cluster_recipe_dict[max_key] + 1

    return cluster_recipe_dict


# function 3
def make_the_order(df, column, n_recipes_in_cluster):

    """Pick the number of recipes from each cluster from the original data base
       according to the customer's choice.
    
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        data containing two columns: cluster and recipe_id and number of rows 
        equal to the existing number of recipes ids
    column: string
         column with recipes ids     
    n_recipes_in_cluster: int
        number of recipes that build up an order

    Returns
    -------
    order: list
        list with recipes ids
            
    """
    df = df.copy()
    clusters = set(df[column].tolist())

    list_order_1 = [
        random.sample(df[column].tolist(), recipe)
        for recipe in n_recipes_in_cluster.values()
    ]
    order_1 = [recipe for sublist in list_order_1 for recipe in sublist]

    for id_ in order_1:
        df = df[df[column] != id_]

    list_order_2 = [
        random.sample(df[column].tolist(), recipe)
        for recipe in n_recipes_in_cluster.values()
    ]
    order_2 = [recipe for sublist in list_order_2 for recipe in sublist]

    logger.info(
        "Orders grouped by clusters:\n"
        "Cheking that number of recipes in each order:\n"
        "is consistent with the output of function 2\n"
        f"{list_order_1}\n"
        f"{list_order_2}\n"
    )

    return order_1, order_2


# data = pd.DataFrame({'id': [i for i in range(n_recipes)], 'C':np.random.choice(clusters, n_recipes, replace=True)})
# customer_choice_df = data[data['id'].isin(ids_chosen)]

# recipes_picked_per_cluster = get_number_of_recipes_per_cluster(customer_choice_df, 'C', n_recipes_in_order)

# logger.info('These are the number of recipes per cluster to pick from the data base:\n'
# f'{recipes_picked_per_cluster}\n')

# recipe_id = 'id'

# logger.info('These are the recipes that make up the order:\n'
# f'{make_the_order(df = data, column = recipe_id, n_recipes_in_cluster = recipes_picked_per_cluster )}')

