import os
from flask import Flask, render_template, request, jsonify, redirect, session
import pandas as pd
import numpy as np
import random
import requests
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from waitress import serve

# from game.utils import run_query
from loguru import logger
from Recipes_Explorer.pca_choices import choose_pca
import sqlite3 as sql

##########################
###### GAME 2  ###########
# It will first ask the food Group, and then depending on the choice, it will ask two question, one for each order.
# The recipes will be chosen from the food group, according to the PCA done.
##########################

counter=0

# create flask instance
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method == "POST":
        
        return redirect("fg_choices")
    
    #customer_id = request.args.get('user')
    #logger.info(f"customer id {customer_id}")
    customer_id = 2
    session["customer_id"]=customer_id

    return render_template("welcome.html")



@app.route("/fg_choices", methods=["GET", "POST"])
def fg_choices():
    error = ""
   
    
    if request.method == "POST":
        # Form being submitted; grab data from form.
        foodGroupChoice = request.form.getlist("choices")

        # Validate form data
        if len(foodGroupChoice) == 0:
            # Form data failed validation; try again
            error = "Please select at least one option"
        else:
            foodGroupChoice = foodGroupChoice[:6] + [0] * (6 - len(foodGroupChoice))
            session["foodGroupChoice"] = foodGroupChoice
            logger.info(f"Food group choice: {foodGroupChoice}")

            return redirect("question1")  # will be first image page


    return render_template("fg_choices.html")


################# Question 1#####################


@app.route("/question1", methods=["GET", "POST"])
def question1():
    error=""
    recipesData = pd.read_csv("../../data/recipe_table.csv", sep=";")

    numRandomChoices = 10
    foodGroupChoice = session["foodGroupChoice"]
    arrayRecipeID = np.zeros((numRandomChoices))
    arrayImageURL = []

    if request.method == "POST":
        logger.info("entered")
        recipeChoice = request.form.getlist("option1")
        if(len(recipeChoice)>=2 and len(recipeChoice)<=5):
            recipeChoice1 = recipeChoice[:numRandomChoices] + [0] * (
                numRandomChoices - len(recipeChoice)
            )
            session["recipeChoice1"] = recipeChoice1
            logger.info(f"Recipes Chosen: {recipeChoice1}")
            return redirect("question2")
        else:
            error="Error- Please choose 2 to 5 recipes"

    numQuestions = 1

    imageDict = {}
    imageDictOne = {}
    imageDictTwo = {}

    # Random generator 3-------
    arrayRecipeID = choose_pca(recipesData, numRandomChoices, foodGroupChoice)

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
        imageDict[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]
        if elem < int(numRandomChoices / 2):
            imageDictOne[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]
        else:
            imageDictTwo[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]

    logger.info(imageDict)

    return render_template(
        "question1.html",
        imageDict=imageDict,
        imageDictOne=imageDictOne,
        imageDictTwo=imageDictTwo,error=error
    )


#####################  Question 2########################


@app.route("/question2", methods=["GET", "POST"])
def question2():
    error=""
    recipesData = pd.read_csv("../../data/recipe_table.csv", sep=";")

    numRandomChoices = 10
    foodGroupChoice = session["foodGroupChoice"]
    arrayRecipeID = np.zeros((numRandomChoices))
    arrayImageURL = []

    if request.method == "POST":
        logger.info("entered")
        recipeChoice = request.form.getlist("option2")
        if(len(recipeChoice)>=2 and len(recipeChoice)<=5):
            recipeChoice2 = recipeChoice[:numRandomChoices] + [0] * (
                numRandomChoices - len(recipeChoice)
            )
            session["recipeChoice2"] = recipeChoice2
            logger.info(f"Recipes Chosen: {recipeChoice2}")

        ##########################################################
        ######### Arya, customer id and necessary calls to the database added  ##########

            #customer_id = 2 # use line below once app is deployed
            customer_id = session["customer_id"]
            logger.info(customer_id)

            all_sessions_string = (session.get("recipeChoice1", None)
                + session.get("recipeChoice2", None)
            )

            all_sessions = [int(float(i)) for i in all_sessions_string]

            all_sessions = [customer_id]+ all_sessions

            logger.info(f"All choices{all_sessions}")

            with sql.connect("database2.db") as con:
                cur = con.cursor()
                cur.executemany(
                    "INSERT INTO tableGame2 (customer_id, o1_c1, o1_c2, o1_c3, o1_c4, o1_c5, o1_c6, o1_c7, o1_c8, o1_c9,\
                            o1_c10, o2_c1, o2_c2, o2_c3, o2_c4, o2_c5, o2_c6, o2_c7, o2_c8, o2_c9, o2_c10) \
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (all_sessions,),
                )
                con.commit()
                msg = "Done"
            # con.close()

            return "Thank you for helping to improve Mindful Chef"
        else:
            error="Error- Please select 2 to 5 recipes"

    imageDict = {}
    imageDictOne = {}
    imageDictTwo = {}

    # Random generator 3-------
    arrayRecipeID = choose_pca(recipesData, numRandomChoices, foodGroupChoice)

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
        imageDict[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]
        if elem < int(numRandomChoices / 2):
            imageDictOne[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]
        else:
            imageDictTwo[elem + 1] = [imageURL, arrayRecipeID[elem], imageRecipeName]

    logger.info(imageDict)
    logger.info("not yet entered")

    return render_template(
        "question2.html",
        imageDict=imageDict,
        imageDictOne=imageDictOne,
        imageDictTwo=imageDictTwo,error=error
    )


# run!
if __name__ == "__main__":
    app.secret_key = "slfdlkgjl"
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=80)
    #serve(app, host='0.0.0.0', port=80)
