import re
import pickle
import numpy as np
import pandas as pd
import gender_guesser.detector as gender
import sys
import os
from nltk.stem.snowball import SnowballStemmer
import time
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas.io.json import json_normalize
from sys import stdout
import logging

logger = logging.getLogger("apiLogger")

detector = gender.Detector()

def preprocess(blob):

	'''

	Preprocesses the JSON data containing user details and the coresponding information on recipes to be suggested.

	'''
	logger.debug("Preprocessing JSON request")
	try:
		#validate the JSON blob before propagating it through the list of functions
		if (len(blob) < 2):
			raise ValueError("The length of JSON data must be more than one")
	except (ValueError) as err:
		logger.exception('Incorectly formatted JSON request: {}'.format(err))
		return('Incorectly formatted JSON request!')                    
	else:
		logger.debug("JSON request is formatted corectly, continuing with processing")
		blob = add_delta(blob)
		blob = add_income(blob)
		blob = add_gender(blob,detector)
		df = dummify(to_dataframe(blob))
		df = imputation(X = df)
		return(df)


def recipe2vec(recipe, merge = True):
	
	'''

	For a given recipe in dictionary format, returns the embedding for its text fields.

	If merge is True, then the seperate embeddings for each text field
	are merged into a full vector.

	(list of floats) <- dict

	'''

	try:
		with open(os.path.dirname(os.path.realpath(__file__)) + "/pretrained/embedder.p", "rb") as f:
			pipelines = pickle.load(f)
	except IOError:
		logger.exception("embedder.p file not found!")
		raise
	recipeText = dict((key,recipe[key]) for key in ['title', 'description', 'key_ingredient']) # extracts text feautures
	recipeText['recipe'] = " ".join(recipeText.values()) # conjoins them
	recipeText = dict((key,[recipeText[key]]) for key in recipeText.keys()) # pipelines expect a list of string(s) to process

	embeddings = dict((key, pipelines[key].transform(recipeText[key])[0]) for key in pipelines.keys()) # gets embedding for each text feature
	
	if merge == True:
	
		mergedembeddings  = np.concatenate(list(embeddings.values())) # merges embeddings
		return mergedembeddings
	
	else:
		
		return embeddings

def calc_delta(embedding1, embedding2):
	
	'''

	Calculates the Euclidean distance between any two embeddings.

	(float) <- list of floats, list of floats

	'''    
	logger.debug("Calculate deltas, embedding1 = {}, embedding2 = {}".format(embedding1, embedding2))
	if (embedding1 is np.nan)|(embedding2 is np.nan):
		
		return np.nan
	
	else:
		delta = np.sqrt(np.sum((embedding1 - embedding2)**2))
		logger.debug("Delta calculated, delta = {}".format(delta))
		return delta


def add_delta(blob):
	
	'''

	This takes as an input a blob and return the same blob
	but with delta1, delta2, and delta12 attached to each suggestion.

	Empty order histories return with nan for the associated delta.

	(dict) <- dict

	'''

	# Checks for existence of previous orders, returns nan for any nonexistent orders, otherwise returns embedding
	logger.debug("Adding deltas")
	if 'lastorder' in blob.keys():


		lastorder_embedding = np.mean([recipe2vec(recipe) for recipe in blob['lastorder']])
		del(blob['lastorder'])

	else:

		lastorder_embedding = np.nan

	if 'lastorder2' in blob.keys():

		lastorder2_embedding = np.mean([recipe2vec(recipe) for recipe in blob['lastorder2']])
		del(blob['lastorder2'])

	else:

		lastorder2_embedding = np.nan

	delta12 = calc_delta(lastorder_embedding, lastorder2_embedding)
	# Get difference between each suggestion and the order histories

	for suggestion in blob['suggestions']:
	   
		embedding = recipe2vec(suggestion)
		
		suggestion['delta1'] = calc_delta(embedding, lastorder_embedding)
		logger.debug("delta1 added, delta1 = {}".format(suggestion['delta1']))  
		suggestion['delta2'] = calc_delta(embedding, lastorder2_embedding)
		logger.debug("delta2 added, delta2 = {}".format(suggestion['delta2']))
		suggestion['delta12'] = delta12
		logger.debug("delta12 added, delta12 = {}".format(suggestion['delta12']))
		del(suggestion['title'])
		del(suggestion['description'])
		del(suggestion['key_ingredient'])
		print(blob)
	return blob

def add_income(blob, deprivationfile = os.path.dirname(os.path.realpath(__file__)) + "/pretrained/postcodedeprivationfinal.csv"):

	'''

	Takes JSON blob with a 'postcode' key and replaces it with an 'IncomeScore' key.

	(dict) <- dict

	'''

	df_deprivation = pd.read_csv(deprivationfile) # Load the deprivation score lookup table
	postcode = blob['user']['postcode'] # Extract postcode of each user
	logger.debug("Adding income. Received post code: {}".format(postcode))
	if df_deprivation[df_deprivation['Postcode'] == postcode].shape[0] > 0:
		income_score = float(df_deprivation[df_deprivation['Postcode'] == postcode]
			['Income Score'].values[0])
		blob['user']['IncomeScore'] = 1.0 - 2*income_score
	else:
		blob['user']['IncomeScore'] = np.NaN
	del blob['user']['postcode']
	logger.debug("Income score of {} added.".format(blob['user']['IncomeScore']))
	return blob


def add_gender(blob, d):

	'''

	Takes blob with a 'first_name' key and replaces it with a dummy variable key 
	'gender_male'.
	d is gender detector to use for gender detection. Gender detector oject contruction was moved out of the function to spead up detection.
	Creating the object takes ±270ms, while deteection itselt takes on ±7ms.
	(dict) <- dict

	'''
	dic = {'mostly_male' : 'male', 'mostly_female' : 'female', 'andy' : np.NaN, 
	'unknown' : np.NaN}
	firstname = blob['user']['first_name'].split(' ')[0]
	logger.debug("Translating name = {} into gender".format(firstname))
 
	g = d.get_gender(firstname)
	g = dic.get(g, g)
	logger.debug("Name {} translated into {} gender".format(firstname, g))
	if g == 'male':
		blob['user']['gender_male'] = 1
	elif g == 'female':
		blob['user']['gender_male'] = 0
	else:
		blob['user']['gender_male'] = np.NaN

	del blob['user']['first_name']
	logger.debug("Gender {} added".format(blob['user']['gender_male']))
	return blob


def to_dataframe(blob):

	'''

	Takes as input blob and turns it into a data frame, one row for each recipe in suggestions.

	(df) <- dict

	''' 
	df = json_normalize(blob['suggestions'])
	#df.rename(columns={'id': 'recipe_id'}, inplace = True)
	df['id'] = blob['user']['id']
	df['IncomeScore'] = blob['user']['IncomeScore']
	df['gender_male'] = blob['user']['gender_male']
	df['meal_plan'] = blob['user']['meal_plan']
	del df['price']
	logger.debug("Converting JSON request to a dataframe: \n {}".format(str(blob)))
	logger.debug("JSON converted to a dataframe: \n {}".format(df.to_string()))
	return df


def dummify(df):

	'''

	Takes a data frame with columns 'meal_plan' and 'food_group' and replaces them with 
	dummy variables.

	(df) <- df

	'''
	logger.debug("Dummifying meal_plan and food_group columns")
	fg_cats = ['Beef', 'Poultry', 'Fish', 'Lamb', 'Pork', 'Vegan', 'Shellfish']
	fg_dummies = pd.get_dummies(df, columns=['food_group'], prefix='', prefix_sep='')
	fg_dummies = fg_dummies.T.reindex(fg_cats).T.fillna(0)
	df = pd.concat([df, fg_dummies], axis=1)

	meal_plan_cats = ['Balanced', 'Pescatarian', 'Plant-Based', 'Protein-Packed']
	mp_dummies = pd.get_dummies(df, columns=['meal_plan'], prefix='', prefix_sep='')
	mp_dummies = mp_dummies.T.reindex(meal_plan_cats).T.fillna(0)
	df = pd.concat([df, mp_dummies], axis=1)

	df = df.drop(['food_group', 'Beef', 'meal_plan', 'Balanced'], axis=1)
	dummy_columns = {
	'Poultry' : 'food_group_Chicken',
	'Fish' : 'food_group_Fish',
	'Lamb' : 'food_group_Lamb',
	'Pork' : 'food_group_Pork',
	'Vegan' : 'food_group_Vegan',
	'Shellfish': 'food_group_Shellfish',
	'Pescatarian' : 'meal_plan_Pescatarian',
	'Plant-Based' : 'meal_plan_Plant-Based',
	'Protein-Packed' : 'meal_plan_Protein-Packed'
	}
	df = df.rename(columns=dummy_columns)
	df = df.drop(['id', 'recipe_id'], axis = 1)
	
	# Sort the columns alphabetically to prep for the imputer

	cols = list(df.columns.values)
	cols.sort()
	df = df[cols]
	logger.debug("Meal_plan and food_group columns dummified. Dataframe updated: \n {}".format(df.to_string()))
	return df

imputer = ""
try:
	with open(os.path.dirname(os.path.realpath(__file__)) + "/pretrained/imputer.p", "rb") as f:
		imputer = pickle.load(f)
except IOError:
	logger.exception("imputer.p file not found!")
	raise


def imputation(X, imputer = imputer):

	'''
	
	For a given (pre-trained) imputer, impute the feature space X.

	Note: The imputer converts to numpy ndarray, so this function maps it back to
	a dataframe with the preserved column names and indices.

	(df) <- df

	'''
	logger.debug("Imputing missing data")
	imputed_DF = pd.DataFrame(imputer.transform(X))
	imputed_DF.columns = X.columns
	imputed_DF.index = X.index
	logger.debug("Missing data imputed. Dataframe updated: \n {}".format(imputed_DF.to_string()))
	return(imputed_DF)


