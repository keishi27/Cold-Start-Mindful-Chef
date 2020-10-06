import os

from flask import Flask, render_template, request, jsonify, redirect, session

import sqlite3 as sql
import pandas as pd
import numpy as np
import random
import requests
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from game.utils import run_query
from loguru import logger
from waitress import serve

from Recipes_Explorer.random_choices import ChooseRandomFoodGroupEqualProb, ChooseRandomEqualProb

############
#### GAME 1
############

numRandomChoices=10
#recipesData=pd.read_csv('C:/Users/lisa/Documents/Mindful_Chef_1/recipe_table.csv', sep=";")
recipesData=pd.read_csv('../../data/recipe_table.csv', sep=";")
recipesData["weight"]=1

####   First asks the food group and then chooses randomly from the food group

# create flask instance
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def intro_page():
    if request.method == "POST":

        return redirect('question1')

    #customer_id = request.args.get('user')
    customer_id = 1
    logger.info(f"customer id {customer_id}")
    session["customer_id"]=customer_id

    return render_template("intro.html")


################# Question 1#####################

@app.route('/question1', methods = ["GET", "POST"])
def question1(numRandomChoices=numRandomChoices):
    error=""

    global recipesData
    
    arrayRecipeID=np.zeros((numRandomChoices))
    arrayImageURL=[]

    # Random generator 1-------
    arrayRecipeID=ChooseRandomEqualProb(recipesData, numRandomChoices)

    for i in arrayRecipeID:
            recipesData.loc[recipesData['id']==i]['weight']=0


    if request.method == "POST":
        logger.info("entered")
        recipeChoice=request.form.getlist('option1')
        if(len(recipeChoice)>=2) and (len(recipeChoice)<=5): 
            recipeChoice1 = recipeChoice[:numRandomChoices] + [0]*(numRandomChoices - len(recipeChoice))
            session["recipeChoice1"] = recipeChoice1
            logger.info(f"Recipes Chosen: {recipeChoice1}")
        
            return redirect('question2') 
        else:
            error="Error- Please choose 2 to 5 recipes."


    

    imageDict={}
    imageDictOne={}
    imageDictTwo={}
   
    logger.info(f"\n array of chosen recipe ID: {arrayRecipeID}")

    for elem in range(numRandomChoices):
        # Getting the image URL, and then splitting it to get the image file name
        imageURL = recipesData.loc[
            recipesData["id"] == arrayRecipeID[elem]
        ].image_url.values[0]

        imageRecipeName = recipesData.loc[
            recipesData["id"] == arrayRecipeID[elem]
        ].title.values[0]
        arrayImageURL.append(imageURL)

        imageDict[elem+1]=[imageURL, arrayRecipeID[elem], imageRecipeName]

        #####   trying to make two dictionaries for displaying in two lines
        if elem < int(numRandomChoices / 2):
            imageDictOne[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]
        else:
            imageDictTwo[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]

    logger.info(imageDict)

    return render_template(
        "question1.html",
        imageDict=imageDict,
        imageDictOne=imageDictOne,
        imageDictTwo=imageDictTwo,error=error)


#####################  Question 2########################

@app.route('/question2', methods = ["GET", "POST"])
def question2(numRandomChoices=numRandomChoices):
    error=""
    global recipesData

    
    arrayRecipeID=np.zeros((numRandomChoices))
    arrayImageURL=[]

    if request.method == "POST":
        logger.info("entered")
        recipeChoice2=request.form.getlist('option2')
        if(len(recipeChoice2)>=2) and (len(recipeChoice2)<=5):
            recipeChoice2 = recipeChoice2[:numRandomChoices] + [0]*(numRandomChoices - len(recipeChoice2))
            session["recipeChoice2"] = recipeChoice2
            logger.info(f"Recipes Chosen: {recipeChoice2}")

            # customer_id = 1 # use line below one app is deployed
            customer_id = session["customer_id"]

            all_sessions_string=session.get("recipeChoice1",None)+session.get("recipeChoice2",None)# replace 1 with url variable
        
            logger.info(f"All Recipes: {all_sessions_string}")
        
            all_sessions = [int(float(i)) for i in all_sessions_string]
            all_sessions = [customer_id]+all_sessions

            with sql.connect("database1.db") as con:
                cur = con.cursor()
                cur.executemany("INSERT INTO tableGame1 (customer_id, que1_rc1, que1_rc2, que1_rc3, que1_rc4, que1_rc5, que1_rc6, que1_rc7, que1_rc8, que1_rc9, que1_rc10, \
                que2_rc1, que2_rc2, que2_rc3, que2_rc4, que2_rc5, que2_rc6, que2_rc7, que2_rc8, que2_rc9, que2_rc10) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (all_sessions,))
                con.commit()
                msg = "Done"
            # con.close()

            return 'Thank you for your help to improve Mindful Chef'
        else:
            error="Error- Please choose 2 to 5 Recipes"

        
    imageDict={}
    imageDictOne={}
    imageDictTwo={}


    # for que in range(numQuestions):

    # Random generator 3-------

    arrayRecipeID=ChooseRandomEqualProb(recipesData, numRandomChoices)
    
    logger.info(f"\n array of chosen recipe ID: {arrayRecipeID}")

    for elem in range(numRandomChoices):
        # Getting the image URL, and then splitting it to get the image file name
        imageURL = recipesData.loc[
            recipesData["id"] == arrayRecipeID[elem]
        ].image_url.values[0]
        imageRecipeName = recipesData.loc[
            recipesData["id"] == arrayRecipeID[elem]
        ].title.values[0]
        arrayImageURL.append(imageURL)

        imageDict[elem+1]=[imageURL, arrayRecipeID[elem], imageRecipeName]
        if(elem<int(numRandomChoices/2)):
            imageDictOne[elem+1]=[imageURL, arrayRecipeID[elem], imageRecipeName]

        else:
            imageDictTwo[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]

    logger.info(imageDict)

    return render_template(
        "question2.html",
        imageDict=imageDict,
        imageDictOne=imageDictOne,
        imageDictTwo=imageDictTwo,error=error)


# run!
if __name__ == "__main__":
    app.secret_key = "slfdlkgjl"
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=80)
    # serve(app, host='0.0.0.0', port=80)
