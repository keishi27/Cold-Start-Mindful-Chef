import pandas as pd
import numpy as np
import random
from loguru import logger

# The three Random generators for recipe ID

def ChooseRandomEqualProb(recipesData, numChoices):
    fullListRecipeId=np.sort(recipesData['id'])

    arrayRecipeID=np.random.choice(fullListRecipeId,numChoices)

    return arrayRecipeID


def ChooseRandomWeightedProb(recipesData, numChoices):

    fullListRecipeId=np.sort(recipesData['id'])
    fullListRatings=np.ones(len(fullListRecipeId))
    
    # Reading Recipe ratings (API).csv
    ratingsDF=pd.read_csv('data/Recipe ratings (API).csv')
    ratingsDF.columns=ratingsDF.columns.str.replace(' ', '_')

    recipeIDSortedList=np.sort(ratingsDF.recipe_id.unique())
    meanSortedRatingsList=np.array(ratingsDF.groupby(['recipe_id']).mean().sort_values(by=['recipe_id']).rating.values)

    for elemindex, elem in enumerate(recipeIDSortedList):
        index=np.where(fullListRecipeId == elem)
        fullListRatings[index]=meanSortedRatingsList[elemindex] # Used as weights

    arrayRecipeID=np.array(random.choices(fullListRecipeId, weights=fullListRatings, k=numChoices))

    return arrayRecipeID


def ChooseRandomFoodGroupWeightedProb(recipesData, numChoices, foodGroupChoice):
    

    # Reading Recipe ratings (API).csv
    ratingsDF=pd.read_csv('data/Recipe ratings (API).csv')
    ratingsDF.columns=ratingsDF.columns.str.replace(' ', '_')

    recipeIDSortedList=np.sort(ratingsDF.recipe_id.unique())
    meanSortedRatingsList=np.array(ratingsDF.groupby(['recipe_id']).mean().sort_values(by=['recipe_id']).rating.values)

    allFoodGroupRecipeID=np.array([])
    allFoodGroupRecipeRatings=np.array([])

    if(numChoices>=len(foodGroupChoice)):
        arrayRecipeID=np.zeros(len(foodGroupChoice))
        counter=0
        for elem in foodGroupChoice:
            foodGroupRecipeID=np.array((recipesData.loc[recipesData['food_group']==elem].id.values))
            

            foodGroupRecipeID=np.sort(foodGroupRecipeID)
            foodGroupRecipeRatings=np.ones(len(foodGroupRecipeID)) # Used as weights
         

            for elemindex, el in enumerate(recipeIDSortedList):
                index=np.where(foodGroupRecipeID == el)
                foodGroupRecipeRatings[index]=meanSortedRatingsList[elemindex]

            arrayRecipeID[counter]=np.array(random.choices(foodGroupRecipeID, weights=foodGroupRecipeRatings, k=1))


            allFoodGroupRecipeID=np.append(allFoodGroupRecipeID,(foodGroupRecipeID), axis=0)
            allFoodGroupRecipeRatings=np.append(allFoodGroupRecipeRatings, foodGroupRecipeRatings, axis=0)

            counter+=1
            
        arrayRecipeID=np.insert(arrayRecipeID, counter, (random.choices(allFoodGroupRecipeID, weights=allFoodGroupRecipeRatings, k=numChoices-counter)))

    else:
        reducedFoodGroupChoice=random.sample(foodGroupChoice, k=numChoices)
        logger.info(f"reduced: {reducedFoodGroupChoice}")
        arrayRecipeID=np.zeros(numChoices)
        counter=0
        for elem in reducedFoodGroupChoice:
            foodGroupRecipeID=np.array((recipesData.loc[recipesData['food_group']==elem].id.values))
                

            foodGroupRecipeID=np.sort(foodGroupRecipeID)
            foodGroupRecipeRatings=np.ones(len(foodGroupRecipeID))
            

            for elemindex, el in enumerate(recipeIDSortedList):
                index=np.where(foodGroupRecipeID == el)
                foodGroupRecipeRatings[index]=meanSortedRatingsList[elemindex] # Used as weights

            arrayRecipeID[counter]=np.array(random.choices(foodGroupRecipeID, weights=foodGroupRecipeRatings, k=1))

            counter+=1



    return arrayRecipeID




