from flask import request, jsonify
import requests
import logging
import json
import datetime
import time
from game.flaskapp_andrius.api.exceptions import (
    NotFoundException,
    AuthenticationException,
    UnauthorizedException,
    BadRequestException,
)

logger = logging.getLogger("apiLogger")


def get_headers():
    """
	Gets headers from the incoming API requests. They are required when making calls to other APIs for getting the data required 
	for making recommendation.
	"""
    headers = {
        "Content-Type": request.headers["Content-Type"],
        "Accept": request.headers["Accept"],
        "Authorization": request.headers["Authorization"],
        "User-Agent": "DataScienceRecipeRecommender/0.1.0",
    }
    return headers


def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


def api_request(headers, api_url, endpoint, parameters):
    try:
        # start_time = time.time()
        response = requests.session().get(
            api_url + endpoint + parameters, headers=headers
        )
        # print('\n query to {} took {}'.format(api_url+endpoint+parameters, time.time() - start_time))
    except requests.exceptions.RequestException as e:
        raise BadRequestException(
            "Unable to get data from {}. \n {}".format(api_url + endpoint, e)
        )
    if response.status_code == 401:
        raise AuthenticationException(
            "Authentication failed. Address: {}. Status code: {}".format(
                api_url + endpoint, response.status_code
            )
        )
    elif response.status_code != 200:
        raise NotFoundException(
            "Unable to get data from {}. Status code: {}".format(
                api_url + endpoint, response.status_code
            )
        )
    else:
        logger.info("Data succesfully retrieved from {}".format(api_url + endpoint))
        return response


def J(*args, **kwargs):
    """Wrapper around jsonify that sets the Content-Type of the response to
	application/vnd.api+json.
	"""
    response = jsonify(*args, **kwargs)
    response.mimetype = "application/vnd.api+json"
    return response


def validate_filters(filters):
    date_format = "%Y-%m-%d"
    meal_plan = ["Balanced", "Pescetarian", "Protein-Packed", "Plant-Based"]
    excluded_food_groups = ["Beef", "Poultry", "Fish", "Lamb", "Pork", "Shellfish"]
    portion_count_per_meal = [1, 2, 4]
    meal_count_per_delivery = {1: [3, 4, 5], 2: [2, 3, 4, 5], 4: [2, 3]}

    if {
        "filter[delivery_date]",
        "filter[meal_plan]",
        "filter[excluded_food_groups]",
        "filter[portion_count_per_meal]",
        "filter[meal_count_per_delivery]",
    } == set(filters.keys()):
        try:
            datetime.datetime.strptime(
                filters.get("filter[delivery_date]"), date_format
            )
        except ValueError:
            raise BadRequestException(
                "Wrong date format received. Date format should be YYYY-MM-DD"
            )

        if filters.get("filter[meal_plan]") not in meal_plan:
            raise BadRequestException(
                "Meal plan has be one of {}. Received: {}".format(
                    meal_plan, filters.get("filter[meal_plan]")
                )
            )

        if len(filters.get("filter[excluded_food_groups]")) > 1:
            for ex_fg in filters.get("filter[excluded_food_groups]").split(","):
                if ex_fg not in excluded_food_groups:
                    raise BadRequestException(
                        "Excluded food groups have to be one of {}. Received: {}".format(
                            excluded_food_groups, ex_fg
                        )
                    )
        try:
            int(filters["filter[portion_count_per_meal]"])
            int(filters["filter[meal_count_per_delivery]"])
        except ValueError:
            raise BadRequestException(
                "Integer values are expected for filters portion_count_per_meal and meal_count_per_delivery."
            )

        if int(filters["filter[portion_count_per_meal]"]) not in portion_count_per_meal:
            raise BadRequestException(
                "Portion count per meal has be one of {}. Received: {}".format(
                    portion_count_per_meal, filters["filter[portion_count_per_meal]"]
                )
            )
        if (
            int(filters["filter[meal_count_per_delivery]"])
            not in meal_count_per_delivery[
                int(filters["filter[portion_count_per_meal]"])
            ]
        ):
            raise BadRequestException(
                "Meal count per delivery has be one of {}. Received: {}".format(
                    meal_count_per_delivery, filters["filter[meal_count_per_delivery]"]
                )
            )

    else:
        raise BadRequestException("Obligatory filters are missing!")

