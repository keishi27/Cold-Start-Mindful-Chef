import pandas as pd
import numpy as np
import random
import requests
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from game.utils import run_query
from loguru import logger
from random_choices import ChooseRandomEqualProb, ChooseRandomWeightedProb, ChooseRandomFoodGroupWeightedProb
from image_display import DisplayImages


# This code explores the recipe_table.csv data file to find the following:
    # 1. All the unique Allergens
    # 2. All the unique cuisines
    # 3. All the unique food groups
    # 4. Whether all the recipes have corresponding image URL
    # 5. Create an array ofrandomly chosen recipe id depending on three algorithms:
        # a. randomly from the entire list with equal probability
        # b. randomly from the entire list with weighted probability, depending on the mean ratings
        # c. randomly from the chosen food groups with the weighted probability
    # 6. Extracts the images from the URL and displays the images
    # 7. Asks for input from user and stores them in a DataFrame
    # 8. The question is repeated a fixed number of times
    


recipesData=pd.read_csv('../../../data/recipe_table.csv', sep=";")
logger.info(recipesData.head())

# to know all the unique allergens
nonzeroAllerAllergens=recipesData['allergens'][recipesData['allergens'].notnull()]


listAllAllergens=[]

for elem in nonzeroAllerAllergens:
    itemsplit=elem.split(',')
    itemstrip=[count.lstrip('{').rstrip('}') for count in itemsplit]
    for item in itemstrip:
        listAllAllergens.insert(0, item)
        
listUniqueAllergens=np.unique(np.array(listAllAllergens))

logger.info(f"List of all unique Allergens:\n {listUniqueAllergens}")

listUniqueCuisines=(recipesData[recipesData['cuisine'].notnull()].cuisine.unique())
logger.info(f"\n List of all unique Cuisines:\n {listUniqueCuisines}")

listUniqueFoodGroups=(recipesData[recipesData['food_group'].notnull()].food_group.unique())
logger.info(f"\n List of all unique food groups:\n {listUniqueFoodGroups}")

logger.info(f"\n Is there any recipe without image URL? {recipesData['image_url'].isnull().any()}")

fullListRecipeId=np.array(recipesData['id'])
logger.info(f"\n Total number of Recipes: {len(fullListRecipeId)}")


# Select the number of random options to be displayed
numRandomChoices=3
arrayRecipeID=np.zeros((numRandomChoices))


# TO DO:
    # 1. Assign to each figure the recipe name, and a number from 1 to numRandomChoices
    # 2. Ask the user to input the choice (e.g. 1-4)
    # 3. Store the corresponding RecipeID in an array.
    # 4. Repeat the entire process depending on how many questions we want to present.
    # 5. Store all the chosen RecipeID in an array for a particular customer

# Using functions

recipe_id_list = []
# numQuestions is number of questions asked
numQuestions=4

for que in range(numQuestions):
    
    # Random generator 1-------
    #arrayRecipeID=ChooseRandomEqualProb(recipesData, numRandomChoices)
    
    # Random generator 2-------
    #arrayRecipeID=ChooseRandomWeightedProb(recipesData, numRandomChoices)
    
    # Random generator 3-------
    foodGroupChoice=['Beef', 'Pork', 'Fish', 'Lamb'] #This should become an input too
    arrayRecipeID=ChooseRandomFoodGroupWeightedProb(recipesData, numRandomChoices, foodGroupChoice)
    
    logger.info(f"\n array of chosen recipe ID: {arrayRecipeID}")
    
    
    DisplayImages(recipesData, arrayRecipeID, numRandomChoices)

    # Catch mistakes in user input
    while True:
        try:
            # user should input 1, 2, 3
            answer = input('Which dish appeals to you the most?')
            option = (int(answer))
            option_index = option-1
            id_store = (arrayRecipeID)[option_index]
        
        except ValueError:
            logger.info(f"Sorry, please input a choice of 1, 2 or 3.")
            # better try again... Return to the start of the loop
            continue
        
        else:
            #choice successfully parsed - exit loop.
            break

    # check option index which should be 1 less than answer/option
    logger.info(f"{option_index}")
    # check id 
    logger.info(f"{id_store}")

    recipe_id_list.append(id_store)

logger.info(f"{recipe_id_list}")

recipe_id_df = pd.DataFrame([recipe_id_list])
logger.info(f"{recipe_id_df}")
