from game.flaskapp_andrius.api import preprocesser
from game.flaskapp_andrius.api.models import (
    Recipe,
    User,
    Recommender_input,
    Order_history,
)
from game.flaskapp_andrius.api.helpers import to_dict
from game.flaskapp_andrius.api.controlers import (
    get_order_history,
    get_order_history_macros,
)
import numpy as np
import os
import logging
import logging.config

# JWT token in Authorization header has to be changed to admin token, making sure we can get order history for all customer ids
headers = {
    "Content-Type": "application/vnd.api+json",
    "Accept": "application/vnd.api+json",
    "Authorization": "Bearer eyJhbGciOiJFRDI1NTE5In0.eyJjdXN0b21lcl9pZCI6IjMxNCIsImF1ZCI6WyJNaW5kZnVsQ2hlZi1BUEkiLCJhdXRoZW50aWNhdGlvbi9sb2dpbl9jcmVkZW50aWFscyIsImNvdXJpZXJzL2F2YWlsYWJpbGl0aWVzIiwiY291cmllcnMvY3VzdG9tZXJfcHJlZmVyZW5jZXMiLCJjb3VyaWVycy9yZWNvbW1lbmRhdGlvbnMiLCJjcmVkaXRfYmFsYW5jZXMvY3VzdG9tZXJfYmFsYW5jZXMiLCJjdXN0b21lcnMvY3VzdG9tZXJzIiwiZGVsaXZlcmllcy9kZWxpdmVyaWVzIiwiZGVsaXZlcmllcy9kZWxpdmVyeV9zdGF0dXNlcyIsImRlbGl2ZXJpZXMvdW5hdmFpbGFibGVfZGF0ZXMiLCJkZWxpdmVyaWVzL2ZlZWRiYWNrcyIsImV4dGVybmFsX3BheW1lbnRzL2NhcmRfYXV0aG9yaXNhdGlvbnMiLCJleHRlcm5hbF9wYXltZW50cy9wYXltZW50X21ldGhvZHMiLCJleHRlcm5hbF9wYXltZW50cy90cmFuc2FjdGlvbnMiLCJwYXltZW50cy90cmFuc2FjdGlvbnMiLCJwYXltZW50cy9wcmV2aWV3cyIsInByb2R1Y3RzL3Byb2R1Y3RzIiwicmVjaXBlcy9hdmFpbGFiaWxpdGllcyIsInJlY2lwZXMvY29tcG9uZW50cyIsInJlY2lwZXMvaW1hZ2VzIiwicmVjaXBlcy9pbmdyZWRpZW50cyIsInJlY2lwZXMvcmVjaXBlcyIsInJlY2lwZXMvcmVjb21tZW5kYXRpb25zIiwicmVjaXBlcy9mZWVkYmFja19yZWFzb25zIiwicmVjaXBlcy9mZWVkYmFja3MiLCJzaG9waWZ5L211bHRpcGFzc191cmxzIiwic3Vic2NyaXB0aW9ucy9mZWVkYmFja3MiLCJzdWJzY3JpcHRpb25zL3N1YnNjcmlwdGlvbnMiLCJ2b3VjaGVycy9kaXNjb3VudF9wcmV2aWV3cyJdLCJleHAiOjE2MDExMzEyMjMsImlzcyI6Ik1pbmRmdWxDaGVmLUFQSSJ9.uCQ8cCHYFlQVxEhIInWJnJN2i4WFrXgO0FbuMRm1VjdQIwkuWmfcTXbhrPEIGJ-7Zo-CIrbdveGNgsd7A0ftAA",
    "User-Agent": "DataScienceRecipeRecommender/0.1.0",
}
"""
logging.config.fileConfig(
    os.path.dirname(os.path.realpath(__file__)) + "/logging.ini",
    disable_existing_loggers=False,
)
logger = logging.getLogger("apiLogger")
"""


def get_delta12(order_hist):
    """
    order_hist is an an instance of a dataclass flaskapp.api.models.Order_history that contains two lists of flaskapp.api.models.Recipe instances, lastorder and lastorder2
    to create this object:
    lastorder = [flaskapp.api.models.Recipe(recipe_id, food_group, calories, carbohydrates, fat, protein, cooking_time, title, description, key_ingredient_description, price)]
    lastorder2 = [flaskapp.api.models.Recipe(recipe_id, food_group, calories, carbohydrates, fat, protein, cooking_time, title, description, key_ingredient_description, price)]
    order_hist = flaskapp.api.models.Order_history(lastorder, lastorder2)

    example of order_hist object follows below:

    {"lastorder": [
            {
                    "recipe_id": "238",
                    "food_group": "Lamb",
                    "calories": "289.1",
                    "carbs": "18.72",
                    "fat": "12.95",
                    "protein": "22.75",
                    "cooking_time": 30,
                    "title": "Creamy lamb korma & broccoli rice",
                    "description": "Broccoli rice is a delicious accompaniment to this creamy lamb spiced korma with cherry tomatoes and mushrooms.",
                    "key_ingredient": "Free-range heritage breed Yorkshire lamb",
                    "price": "No price"
            },
            {
                    "recipe_id": "419",
                    "food_group": "Vegan",
                    "calories": "672.93",
                    "carbs": "97.52",
                    "fat": "14.71",
                    "protein": "39.09",
                    "cooking_time": 30,
                    "title": "Sticky tamarind & ginger tofu with mangetout",
                    "description": "Shiitake mushrooms are absolutely brimming with umami and make this tangy tofu dish completely irresistible. The tamarind and fresh aromatics add even more flavour.",
                    "key_ingredient": "Dragonfly organic extra-firm tofu ",
                    "price": "No price"
            }
    ],
    "lastorder2": [
            {
                    "recipe_id": "419",
                    "food_group": "Vegan",
                    "calories": "672.93",
                    "carbs": "97.52",
                    "fat": "14.71",
                    "protein": "39.09",
                    "cooking_time": 30,
                    "title": "Sticky tamarind & ginger tofu with mangetout",
                    "description": "Shiitake mushrooms are absolutely brimming with umami and make this tangy tofu dish completely irresistible. The tamarind and fresh aromatics add even more flavour.",
                    "key_ingredient": "Dragonfly organic extra-firm tofu ",
                    "price": "No price"
            },
            {
                    "recipe_id": "532",
                    "food_group": "Fish",
                    "calories": "818.8",
                    "carbs": "86.64",
                    "fat": "39.61",
                    "protein": "29.09",
                    "cooking_time": 0,
                    "title": "Haddock, Asian veg & coconut black rice ",
                    "description": "Coconut black rice infused with ginger, garlic and chilli contrasts with delicate white haddock and a selection of crisp Asian vegetables. As always, our fish is responsibly sourced. The perfect meal for date night? \n",
                    "key_ingredient": "Fresh, sustainably caught haddock fillet",
                    "price": "No price"
            }
    ]}


    """
    if "lastorder" in order_hist.keys():
        lastorder_embedding = np.mean(
            [preprocesser.recipe2vec(recipe) for recipe in order_hist["lastorder"]]
        )
    else:
        lastorder_embedding = np.nan

    if "lastorder2" in order_hist.keys():
        lastorder2_embedding = np.mean(
            [preprocesser.recipe2vec(recipe) for recipe in order_hist["lastorder2"]]
        )
    else:
        lastorder2_embedding = np.nan

    delta12 = preprocesser.calc_delta(lastorder_embedding, lastorder2_embedding)

    return delta12


def run_preprocessing_pipeline(order_hist):
    """
    this function wrapper should only used for testing purposes. Instead of running only functions required for delta12 calculation as in get_delta12(order_hist),
    it runs entire preprocessing pipeline generating input for the recipe recommender. For simplicity, I removed all unnecesary stuff and made a wrapper around
    proprocessing pipeline. This wrapper function takes exactly the same parameter as get_delta12(order_hist) and returns only delta12 value. However more
    computations are performend under the hood that are not required for the onboarding tool.
    """
    # hardcoded extra parameters that will not change during the experiments
    customer = User(id=1, first_name="John", meal_plan="Balanced", postcode="SW18")
    suggestions = [
        Recipe(
            recipe_id="1",
            food_group="Pork",
            calories="150",
            carbs="23",
            fat="50",
            protein="400",
            cooking_time="25",
            title="Meal 1",
            description="description of meal 1",
            key_ingredient="key ingredient 1",
            price="900",
        )
    ]

    order_hist = Recommender_input(
        user=customer,
        suggestions=suggestions,
        lastorder=order_hist["lastorder"],
        lastorder2=order_hist["lastorder2"],
    )
    delta12 = preprocesser.preprocess(to_dict(order_hist))["delta12"][0]

    return delta12


def get_delta12_api(headers, customer_id):
    """
    This funtion will get delta12 value for the "real" order history given coustomer ID. It collects data from Mindful Chef production database and uses
    get_delta12(order_hist) function to calculate the actual value of delta12.
    """
    orders = get_order_history(headers, customer_id)
    order_history_macros = get_order_history_macros(orders, 2, headers)
    delta12 = get_delta12(
        to_dict(
            Order_history(
                lastorder=order_history_macros[0], lastorder2=order_history_macros[1]
            )
        )
    )

    return delta12


def get_recipe_id_api(headers, customer_id):
    """
    This funtion will get recipe IDs for the "real" order history given coustomer ID. It collects data from Mindful Chef production database.
    """
    orders = get_order_history(headers, customer_id)
    # order_history_macros = get_order_history_macros(orders, 2, headers)

    return orders


def get_recipe_api(headers, customer_id):
    """
    This funtion will get recipe IDs for the "real" order history given coustomer ID. It collects data from Mindful Chef production database.
    """
    orders = get_order_history(headers, customer_id)
    order_history_macros = get_order_history_macros(orders, 2, headers)

    return order_history_macros


def get_text_vector(recipe):
    """
    This function outputs a 45-dimensional vector for recipe text fields (title, description and key_ingredient)
    """
    return preprocesser.recipe2vec(recipe)


if __name__ == "__main__":
    # two example orders for testing
    lastorder = [
        Recipe(
            recipe_id="1",
            food_group="Pork",
            calories="150",
            carbs="23",
            fat="50",
            protein="400",
            cooking_time="25",
            title="Meal 1",
            description="description of meal 1",
            key_ingredient="key ingredient 1",
            price="900",
        ),
        Recipe(
            recipe_id="2",
            food_group="Fish",
            calories="200",
            carbs="45",
            fat="100",
            protein="500",
            cooking_time="30",
            title="Meal 2",
            description="description of meal 2",
            key_ingredient="key ingredient 2",
            price="800",
        ),
    ]
    lastorder2 = [
        Recipe(
            recipe_id="1",
            food_group="Pork",
            calories="150",
            carbs="23",
            fat="50",
            protein="400",
            cooking_time="25",
            title="Meal 1",
            description="description of meal 1",
            key_ingredient="key ingredient 1",
            price="900",
        ),
        Recipe(
            recipe_id="2",
            food_group="Fish",
            calories="200",
            carbs="45",
            fat="100",
            protein="500",
            cooking_time="30",
            title="Meal 2",
            description="description of meal 2",
            key_ingredient="key ingredient 2",
            price="800",
        ),
    ]
    # example recipe
    recipe1 = Recipe(
        recipe_id="1",
        food_group="Pork",
        calories="150",
        carbs="23",
        fat="50",
        protein="400",
        cooking_time="25",
        title="Meal 1",
        description="description of meal 1",
        key_ingredient="key ingredient 1",
        price="900",
    )

    # print(get_delta12(to_dict(Order_history(lastorder = lastorder, lastorder2 = lastorder2))))
    # print(run_preprocessing_pipeline(to_dict(Order_history(lastorder = lastorder, lastorder2 = lastorder2))))
    print(get_delta12_api(headers, 314))
    # print(get_text_vector(to_dict(recipe1)))
