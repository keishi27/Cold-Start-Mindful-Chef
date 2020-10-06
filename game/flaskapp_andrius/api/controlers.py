from game.flaskapp_andrius.config.api_config import (
    DATA_API_URL,
    RECIPE_AVAILABILITY_ENDPOINT,
    RECIPE_MACROS_ENDPOINT,
    CUSTOMER_ENDPOINT,
    SUBSCRIPTIONS_ENDPOINT,
    DELIVERIES_ENDPOINT,
)
from game.flaskapp_andrius.api.helpers import api_request
from game.flaskapp_andrius.api.exceptions import NotFoundException
from game.flaskapp_andrius.api import models
import logging

FOOD_GROUP = ["Beef", "Poultry", "Fish", "Lamb", "Pork", "Shellfish"]
# in case none of these food groups is present, the recipe is "Vegan"

logger = logging.getLogger("apiLogger")


def get_available_recipes(
    delivery_date, portion_count_per_meal, excluded_food_groups, headers
):

    """ returns a list of models.Recipe objects containing recipe macros
	[{
		"id": "426",
		"food_group": "foodgroup",
		"calories": "1028.67",
		"carbs": "53.1",
		"fat": "35.05",
		"protein": "96.49",
		"cooking_time": "cooking time",
		"title": "Pink peppercorn pork, sticcoli & cashews",
		"description": "Tangy sweet pink peppercorns are stir-fried with sticcoli, succulent slices of pork and crunchy cashew nuts. ",
		"key_ingredient": "key ingrediant",
		"price": "15.0"
	}]

	"""

    excluded_f_g = excluded_food_groups.split(",")
    filters = "?filter[available_from][lte]={}&filter[available_until][gte]={}&filter[portions][eq]={}&include=recipe,recipe.macros_per_person_for_{}_portion,recipe.steps,recipe.ingredients&fields[availabilities]=recipe_id,price&fields[recipes]=id,title,description,key_ingredient_description&fields[ingredients]=id,food_group&fields[recipe_steps]=id,duration_minutes".format(
        delivery_date, delivery_date, portion_count_per_meal, portion_count_per_meal
    )
    response = api_request(headers, DATA_API_URL, RECIPE_AVAILABILITY_ENDPOINT, filters)
    recipe_ids = []
    available_recipes = []
    if len(response.json()["data"]) > 0:
        for recipe in response.json()["data"]:
            try:
                filtered = list(
                    filter(
                        lambda x: x["id"] == recipe["attributes"]["recipe_id"],
                        response.json()["included"],
                    )
                )
                step_ids = [
                    int(d["id"]) for d in filtered[0]["relationships"]["steps"]["data"]
                ]
                cooking_time = sum(
                    int(d["attributes"]["duration_minutes"])
                    for d in response.json()["included"]
                    if int(d["id"]) in step_ids and d["attributes"]["duration_minutes"]
                )

                ingredient_ids = [
                    int(d["id"])
                    for d in filtered[0]["relationships"]["ingredients"]["data"]
                ]
                ingredient_food_groups = [
                    (d["attributes"]["food_group"])
                    for d in response.json()["included"]
                    if int(d["id"]) in ingredient_ids and d["attributes"]["food_group"]
                ]
                f_g = set.intersection(set(ingredient_food_groups), set(FOOD_GROUP))
                if len(f_g) == 0:
                    food_group = "Vegan"
                elif len(f_g) == 1:
                    food_group = f_g.pop()
                else:
                    raise NotFoundException(
                        "Error occured when retrieving available recipes. Impossible to determine food group for recipe id = {}. Overlap between expected and identified food_group values is {}".format(
                            recipe["attributes"]["recipe_id"], f_g
                        )
                    )
                if food_group not in excluded_f_g:
                    recipe = models.Recipe(
                        recipe["attributes"]["recipe_id"],
                        food_group,
                        filtered[1]["attributes"]["calories"],
                        filtered[1]["attributes"]["carbohydrates"],
                        filtered[1]["attributes"]["fat"],
                        filtered[1]["attributes"]["protein"],
                        cooking_time,
                        filtered[0]["attributes"]["title"],
                        filtered[0]["attributes"]["description"],
                        filtered[0]["attributes"]["key_ingredient_description"],
                        recipe["attributes"]["price"],
                    )
                    available_recipes.append(recipe)
            except IndexError:
                logger.error(
                    "Error occured when retrieving available recipes. Recipe_id = {} lacks necessary data. Data retrieved from {}".format(
                        recipe["attributes"]["recipe_id"],
                        DATA_API_URL + RECIPE_AVAILABILITY_ENDPOINT + filters,
                    )
                )
        if len(available_recipes) < 1:
            raise NotFoundException(
                "Error occured when retrieving available recipes. No recipes meeting the criteria for excluded food groups available!. Exluded food groups: {}".format(
                    excluded_food_groups
                )
            )
        else:
            return available_recipes
    else:
        raise NotFoundException(
            "Error occured when retrieving available recipes. No available recipes found!"
        )


def get_customer_properties(meal_plan, headers, customer_id):
    """ returns user properties required for making a a recomemndation. Meal plan is set from the GET request, not taken from customer details:
	{
	"id": "314",
	"first_name": "Andrius",
	"meal_plan": "Pescetarian",
	"postcode": "E1"
	}
	"""
    # filters = '?include=default_delivery_address,meal_preferences&fields[customers]=id,first_name&fields[default_delivery_addresses]=postcode'
    filters = "?filter[id][eq]={}&include=delivery_address&fields[subscriptions]=customer_id&fields[delivery_addresses]=recipient_name,postcode".format(
        customer_id
    )
    response = api_request(headers, DATA_API_URL, SUBSCRIPTIONS_ENDPOINT, filters)
    customer = models.User(
        response.json()["data"][0]["attributes"]["customer_id"],
        response.json()["included"][0]["attributes"]["recipient_name"],
        meal_plan,
        response.json()["included"][0]["attributes"]["postcode"].split(" ")[0],
    )

    return customer


def get_order_history(headers, customer_id):
    """
	returns order history of two last orders each containing a list of recipe ids
	[{'3339': ['419', '238']}, {'3338': ['591', '372']}]

	"""

    filters = "?include=contents&filter[status][eq]=billed&filter[customer_id][eq]={}&sort=-delivery_date&fields[deliveries]=id&fields[contents]=recipe_id&page[size]=2".format(
        customer_id
    )
    response = api_request(headers, DATA_API_URL, DELIVERIES_ENDPOINT, filters)
    deliveries = []
    # get two last delivery ids use them to find their content ids. Use content ids to find recipe ids
    if len(response.json()["data"]) > 1:
        for i in range(2):
            content_ids = []
            for content in response.json()["data"][i]["relationships"]["contents"][
                "data"
            ]:
                content_ids.append(
                    list(
                        filter(
                            lambda x: x["id"] == content["id"],
                            response.json()["included"],
                        )
                    )[0]["attributes"]["recipe_id"]
                )
            deliveries.append({response.json()["data"][i]["id"]: content_ids})
    else:
        raise NotFoundException(
            "Customer order history is too short! Two previuos orders required to make a recommendation!"
        )
    return deliveries


def get_order_history_macros(order_history, portion_count_per_meal, headers):
    """
	order_history is a list of two last orders each containing a list of recipe in an order:
	[{'3339': ['419', '238']}, {'3338': ['591', '372']}]

	portion_count_per_meal - int

	returns a list of models.Recipe objects representing lastorder and lastorder2
[
	[
		{
			"id": "72",
			"food_group": "food group",
			"calories": "463.27",
			"carbs": "94.39",
			"fat": "4.5",
			"protein": "13.48",
			"cooking_time": "30",
			"title": "Butternut squash & chicken quinoa risotto",
			"description": "A delicious roasted squash, chicken & parsley risotto using British quinoa instead of white rice",
			"key_ingredient": "Dragonfly organic extra-firm tofu",
			"price": "No price"
		}
	],
	[
		{
			"id": "72",
			"food_group": "food group",
			"calories": "463.27",
			"carbs": "94.39",
			"fat": "4.5",
			"protein": "13.48",
			"cooking_time": "30",
			"title": "Butternut squash & chicken quinoa risotto",
			"description": "A delicious roasted squash, chicken & parsley risotto using Britisgh quinoa instead of white rice",
			"key_ingredient": "Dragonfly organic extra-firm tofu",
			"price": "No price"
		}
	]
]

"""
    lastorder_ids = list(order_history[0].values())[0]
    lastorder2_ids = list(order_history[1].values())[0]
    recipe_ids = ",".join(lastorder_ids) + "," + ",".join(lastorder2_ids)

    filters = "?include=macros_per_person_for_{}_portion,steps.recipe,ingredients&filter[id]={}".format(
        portion_count_per_meal, recipe_ids
    )
    response = api_request(headers, DATA_API_URL, RECIPE_MACROS_ENDPOINT, filters)

    if len(response.json()["data"]) >= 2:
        return [
            populate_previuos_order(lastorder_ids, response),
            populate_previuos_order(lastorder2_ids, response),
        ]
    else:
        raise NotFoundException(
            "Recipe macros cannot be retrieved. Order history must contain at least two distinct recipes. Recipe IDs: {}".format(
                recipe_ids
            )
        )


def populate_previuos_order(recipe_ids, response):
    lastorder = []
    for recipe_id in recipe_ids:
        try:
            filtered_data = list(
                filter(lambda x: x["id"] == recipe_id, response.json()["data"])
            )
            filtered_include = list(
                filter(lambda x: x["id"] == recipe_id, response.json()["included"])
            )
            step_ids = [
                int(d["id"]) for d in filtered_data[0]["relationships"]["steps"]["data"]
            ]
            cooking_time = sum(
                int(d["attributes"]["duration_minutes"])
                for d in response.json()["included"]
                if int(d["id"]) in step_ids and d["attributes"]["duration_minutes"]
            )
            ingredient_ids = [
                int(d["id"])
                for d in filtered_data[0]["relationships"]["ingredients"]["data"]
            ]
            ingredient_food_groups = [
                (d["attributes"]["food_group"])
                for d in response.json()["included"]
                if int(d["id"]) in ingredient_ids and d["attributes"]["food_group"]
            ]
            f_g = set.intersection(set(ingredient_food_groups), set(FOOD_GROUP))
            if len(f_g) == 0:
                food_group = "Vegan"
            elif len(f_g) == 1:
                food_group = f_g.pop()
            else:
                raise NotFoundException(
                    "Error occured when retrieving order history. Impossible to determine food_group for recipe id = {}. Overlap between expected and identified food_group values is {}".format(
                        recipe["attributes"]["recipe_id"], f_g
                    )
                )

            last_o = models.Recipe(
                recipe_id,
                food_group,
                filtered_include[0]["attributes"]["calories"],
                filtered_include[0]["attributes"]["carbohydrates"],
                filtered_include[0]["attributes"]["fat"],
                filtered_include[0]["attributes"]["protein"],
                cooking_time,
                filtered_data[0]["attributes"]["title"],
                filtered_data[0]["attributes"]["description"],
                filtered_data[0]["attributes"]["key_ingredient_description"],
                "No price",
            )
            lastorder.append(last_o)
        except IndexError:
            raise NotFoundException(
                "Error occured when retrieving order history. Recipe_id = {} lacks necessary macros data.".format(
                    recipe_id
                )
            )

    return lastorder

