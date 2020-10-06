import json
from dataclasses import dataclass
from typing import Dict
import logging
import sys
from marshmallow import validate, ValidationError  
from marshmallow_jsonapi import fields  
from marshmallow_jsonapi.flask import Relationship, Schema 

logger = logging.getLogger("apiLogger")


FOOD_GROUP = ["Beef", "Poultry", "Fish", "Lamb", "Pork", "Shellfish", "Vegan"]
MEAL_PLAN = ["Balanced", "Pescetarian", "Plant-Based", "Protein-Packed"]

### Data schema for the recommedner input ###
@dataclass
class Recipe:
	recipe_id: str
	food_group: str
	calories: str
	carbs: str
	fat: str
	protein: str
	cooking_time: str
	title: str
	description: str
	key_ingredient: str
	price: str
	
	#Uncomment for production! There is an error in staging data, trigerring validation error!
	'''
	def __post_init__(self):
		try:
			if self.food_group not in FOOD_GROUP:
				raise ValueError("Values of food_group has to be in {}. Received {}. Recipe ID = {}".format(FOOD_GROUP, self.food_group, self.recipe_id))
			elif float(self.calories) <= 0:
				raise ValueError("Calorie count should be a positive value. Received {}. Recipe ID = {}".format(self.calories, self.recipe_id))
			elif float(self.carbs) <= 0:
				raise ValueError("Carbohydrate should be a positive value. Received {}. Recipe ID = {}".format(self.carbs, self.recipe_id))
			elif float(self.fat) <= 0:
				raise ValueError("Fat should be a positive value. Received {}. Recipe ID = {}".format(self.fat, self.recipe_id))
			elif float(self.protein) <= 0:
				raise ValueError("Protein should be a positive value. Received {}. Recipe ID = {}".format(self.protein, self.recipe_id))
			elif float(self.cooking_time) <= 0:
				raise ValueError("Cooking_time should be a positive value. Received {}. Recipe ID = {}".format(self.cooking_time, self.recipe_id))
			elif float(self.price) <= 0:
				raise ValueError("Price should be a positive value. Received {}. Recipe ID = {}".format(self.price, self.recipe_id))
			elif len(self.title) < 1:
				raise ValueError("Recipe title is missing. Recipe ID = {}".format(self.recipe_id))
			elif len(self.description) < 1:
				raise ValueError("Recipe description is misisng. Recipe ID = {}".format(self.recipe_id))
			elif len(self.description) < 1:
				raise ValueError("Recipe description is missing. Recipe ID = {}".format(self.recipe_id))
		except (ValueError) as err:
			logger.exception("Incorrect data for making a recommendation received from data APIs: {}".format(err))
			#return('Incorectly formatted JSON request!')   
			raise ValueError("Incorrect data for making a recommendation received from data APIs: {}".format(err))
	'''

@dataclass
class User:
	id: int
	first_name: str
	meal_plan: str
	postcode: str

	def __post_init__(self):
		try:
			if len(self.first_name) < 1:
				raise ValueError("Customer name is missing, customer ID = {}".format(self.id))
			elif self.meal_plan not in MEAL_PLAN:
				raise ValueError("Values of meal_plan has to be in {}. Received {}. Customer ID = {}".format(MEAL_PLAN, self.meal_plan, self.id))
			elif type(self.postcode) is not str:
				raise TypeError("Field 'postcode' must be of type 'str'.")
		except(TypeError, ValueError) as err:
			logger.exception('Incorrect customer data received: {}'.format(err))
			raise ValueError("Incorrect data for making a recommendation received from data APIs: {}".format(err))


@dataclass
class Recommender_input:
	user: User
	suggestions: Dict[str, Recipe]
	lastorder: Dict[str, Recipe]
	lastorder2: Dict[str, Recipe]

	def toJSON(self):
		return json.dumps(self, default=lambda o: o.__dict__, indent=4)

@dataclass
class Order_history:
	lastorder: Dict[str, Recipe]
	lastorder2: Dict[str, Recipe]


### Data schema for JSON:API output ###
class RecipeRecommendations(Schema):
	id = fields.Str(dump_only=True)
	title = fields.Str()
	recipe_id = fields.Str()
	portions = fields.Int()
	price = fields.Str()
	priority = fields.Int()

	class Meta:
		type_ = "recommendations"
		#self_view = "recommendation_detail"
		#self_view_kwargs = {"customer_id": "<customer_id>", "_external": True}
		#self_view_many = "recommendation_list"
		#self_view_many_kwargs = {"customer_id": "<customer_id>", "_external": True}


### Data schema for JSON:API erors ###

class Error(Schema):
	id = fields.Str(dump_only=True)
	code = fields.Str()
	status = fields.Str()
	title = fields.Str()
	detail = fields.Str()

	class Meta:
		type_ = "recommendations_error"


