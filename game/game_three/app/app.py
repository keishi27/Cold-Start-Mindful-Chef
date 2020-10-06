
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
from loguru import logger
from Recipes_Explorer.game_three_functions import get_equal_recipe_id_from_clusters, get_number_of_recipes_per_cluster, make_the_order
from waitress import serve

########################################
####### GAME 3 #######
# Valentina and Giovanni #########
n_of_clusters=4
n_recipes_per_cluster= 6
n_questions = 3
n_recipes_in_order = 5
n_options_per_questions = int(n_of_clusters*n_recipes_per_cluster/n_questions) 

topicModelDF=pd.read_csv('../../data/df_game.csv', sep=",")
columnCluster="cluster"

#fullListRecipes=get_equal_recipe_id_from_clusters(topicModelDF, columnCluster, n_of_clusters, n_recipes_per_cluster )

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def intro_page():
    if request.method == "POST":
        return redirect("question1")
    #customer_id = request.args.get('user')
    #logger.info(f"customer id {customer_id}")
    customer_id = 2 
    session["customer_id"]=customer_id
    fullListRecipes=get_equal_recipe_id_from_clusters(topicModelDF, columnCluster, n_of_clusters, n_recipes_per_cluster )
    session["fullListRecipes"] = fullListRecipes
    return render_template("intro.html")

###############  Question 1######################
#@app.route('/', methods=["GET", "POST"])
@app.route('/question1', methods=["GET", "POST"])
def question1():
    error = ""
    fullListRecipes = session["fullListRecipes"]
    logger.info(f"full list of recipes:{fullListRecipes}")
    
    if request.method == "POST":
        # Form being submitted; grab data from form.
        recipeChoice = request.form.getlist('option1')
        if(len(recipeChoice)>=2): 
            session["recipeChoice1"] = recipeChoice
            #session["fullListRecipes"]=fullListRecipes

            logger.info(f"1st choices made: {recipeChoice}")
        
            return redirect('question2') # will be the second question

        else:
            error= "Error- Choose at least 2 to 8 recipes"

    
    recipesData=pd.read_csv('../../data/recipe_table.csv', sep=";")

    arrayRecipeID=np.zeros((n_options_per_questions))    

    for i in range(n_of_clusters):
        arrayRecipeID[i*2]=fullListRecipes[i][0]
        arrayRecipeID[i*2+1]=fullListRecipes[i][1]

    #arrayRecipeID[4]=fullListRecipes[0][4]
    logger.info(f"Options shown in Q1: {arrayRecipeID}")

    imageDict={}
    for elem in range(n_options_per_questions):
        # Getting the image URL, and then splitting it to get the image file name
        imageURL=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].image_url.values[0]
        imageRecipeName=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].title.values[0]

        imageDict[elem+1]=[imageURL, arrayRecipeID[elem], imageRecipeName]


    return render_template('question1.html', imageDict=imageDict, error=error)




################# Question 2#####################

@app.route('/question2', methods = ["GET", "POST"])
def question2():


    fullListRecipes = session["fullListRecipes"]
    error = ""
    

    if request.method == "POST":
        # Form being submitted; grab data from form.
        recipeChoice = request.form.getlist('option2')
        if(len(recipeChoice)>=2):
            session["recipeChoice2"] = recipeChoice
            logger.info(f"2nd choices made: {recipeChoice}")
            return redirect('question3') # will be the third question

        else:
            error= "Error- Choose at least 2 to 8 recipes"


    recipesData=pd.read_csv('../../data/recipe_table.csv', sep=";")

    arrayRecipeID=np.zeros((n_options_per_questions))

    for i in range(n_of_clusters):
        arrayRecipeID[2*i]=fullListRecipes[i][2]
        arrayRecipeID[2*i+1]=fullListRecipes[i][3]
    #arrayRecipeID[4]=fullListRecipes[1][4]
    logger.info(f"Options shown in Q2: {arrayRecipeID}")

    imageDict={}
    for elem in range(n_options_per_questions   ):
        # Getting the image URL, and then splitting it to get the image file name
        imageURL=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].image_url.values[0]
        imageRecipeName=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].title.values[0]

        imageDict[elem+1]=[imageURL, arrayRecipeID[elem], imageRecipeName]


    return render_template('question2.html', imageDict=imageDict, error=error)
    

#####################  Question########################

'''@app.route('/question3', methods = ["GET", "POST"])
def question3():
    
    fullListRecipes = session["fullListRecipes"]
    
    error = ""

    if request.method == "POST":
        # Form being submitted; grab data from form.
        recipeChoice = request.form.getlist('option3')
        if(len(recipeChoice)>=2):
            session["recipeChoice3"] = recipeChoice
            logger.info(f"3rd choices made: {recipeChoice}")
            return redirect('question4') # will be the fourth question

        else:
            error="Error- Choose at least one"


    recipesData=pd.read_csv('../../data/recipe_table.csv', sep=";")

    
    arrayRecipeID=np.zeros((n_options_per_questions))


    for i in range(n_of_clusters):
        arrayRecipeID[2*i]=fullListRecipes[i][4]
        arrayRecipeID[2*i+1]=fullListRecipes[i][5]

    #arrayRecipeID[4]=fullListRecipes[2][4]
    logger.info(f"Options shown in Q3: {arrayRecipeID}")

    imageDict={}
    for elem in range(n_recipes_per_cluster):
        # Getting the image URL, and then splitting it to get the image file name
        imageURL=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].image_url.values[0]
        imageRecipeName=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].title.values[0]

        imageDict[elem+1]=[imageURL, arrayRecipeID[elem], imageRecipeName]
        


    return render_template('question3.html', imageDict=imageDict, error=error)'''
    


#################### Question 3 ################################

@app.route('/question3', methods = ["GET", "POST"])
def question3():
    #recipeChoiceList = session["recipeChoice"]
    fullListRecipes = session["fullListRecipes"]
    error = ""

    if request.method == "POST":
        # Form being submitted; grab data from form.
        recipeChoice = request.form.getlist('option3')
        if(len(recipeChoice)>=2):
            session["recipeChoice3"] = recipeChoice
            logger.info(f"3rd choices made: {recipeChoice}")
            topicModelDF=pd.read_csv('../../data/df_game.csv', sep=",")
            index=0
            dictRecipesChosen={}
            clusterList=np.array([])
            recipeChoiceListString=session["recipeChoice1"]+session["recipeChoice2"]+session["recipeChoice3"]
            recipeChoiceList=[int(float(recipeChoiceListString[i])) for i in range(len(recipeChoiceListString))]
            logger.info(f"All chosen recipes:{recipeChoiceList}")
            
            for elem in recipeChoiceList:
                clusterList=np.append(clusterList, topicModelDF.loc[topicModelDF['id']==elem].cluster.values)
        
            data={'cluster':clusterList, 'id':recipeChoiceList}
            chosenDataFrame=pd.DataFrame(data)
        
            columnCluster="cluster"
        
            dictClusterValue=get_number_of_recipes_per_cluster(chosenDataFrame, columnCluster, n_recipes_in_order)
        
        
            columnID="id"
            orderOne, orderTwo=make_the_order(topicModelDF, columnID, dictClusterValue) 
            logger.info(f"Order One: {orderOne}")
            logger.info(f"Order Two: {orderTwo}")
            
        
        ######################## Writing to the Database  #######################
        
            #customer_id = 3 # line below will be used once app is deployed
            customer_id = session["customer_id"] 
            allOrders= [customer_id]+orderOne+orderTwo
        
            with sql.connect("database3.db") as con:
                cur = con.cursor()
                cur.executemany("INSERT INTO tableGame3 (customer_id, que1_rc1, que1_rc2, que1_rc3, que1_rc4, que1_rc5, \
                que2_rc1, que2_rc2, que2_rc3, que2_rc4, que2_rc5) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (allOrders,))
                con.commit()
                msg = "Done"

        

            return 'Thank you for your help to improve Mindful Chef'

        else:
            error="Error- Choose at least 2 to 8 recipes"


    recipesData=pd.read_csv('../../data/recipe_table.csv', sep=";")

   

    arrayRecipeID=np.zeros((n_options_per_questions))


    for i in range(n_of_clusters):
        arrayRecipeID[2*i]=fullListRecipes[i][4]
        arrayRecipeID[2*i+1]=fullListRecipes[i][5]
    logger.info(f"Options shown in Q3: {arrayRecipeID}")

    imageDict={}
    for elem in range(n_options_per_questions):
        # Getting the image URL, and then splitting it to get the image file name
        imageURL=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].image_url.values[0]
        imageRecipeName=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].title.values[0]

        imageDict[elem+1]=[imageURL, arrayRecipeID[elem], imageRecipeName]


    return render_template('question3.html', imageDict=imageDict, error=error)


# run!
if __name__ == '__main__':
    app.secret_key = "slfdlkgjl"
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=80)
    #serve(app, host='0.0.0.0', port=80)
