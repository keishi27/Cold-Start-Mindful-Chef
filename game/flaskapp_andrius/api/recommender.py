import pickle
import pandas as pd
import numpy as np
import datetime
import time

from flaskapp.api import ranker
from flaskapp.api import preprocesser
import uuid 


def recommender(blob, portion_count_per_meal):

	'''

	High level recommendation function chaining preprocessing and prediction.

	outputs recommendations in a list of dicts format:
	[{'id': UUID('b3bf2326-6da3-11ea-9e44-acde48001122'), 'recipe_id': '1', 'portions': '2', 'price': '10.0', 'priority': 11}]

	'''
	df = preprocesser.preprocess(blob)
	ranks = [int(x) for x in ranker.rankrecipes(df)]
	results = []
	for suggestion, rank in zip(blob['suggestions'], ranks):
		results.append({'id': uuid.uuid1(), 'recipe_id': suggestion['recipe_id'], 'portions': portion_count_per_meal, 'price': suggestion['price'], 'priority': rank})
	return(results)

	
