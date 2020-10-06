from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flaskapp.api import recommender
from flaskapp.api import models
from flaskapp.config.api_config import PUBLIC_KEY, AUDIENCE
from flaskapp.api.helpers import get_headers, to_dict, J, validate_filters
from flaskapp.api.controlers import get_available_recipes, get_customer_properties, get_order_history, get_order_history_macros 
from flaskapp.api.exceptions import NotFoundException, AuthenticationException, UnauthorizedException, BadRequestException

import uuid 
import os
import sys
from functools import wraps
import json
import time
from multiprocessing.pool import ThreadPool

import jwt
import nacl.signing
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import logging
import logging.config





app = Flask(__name__)
api = Api(app)


logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = logging.getLogger("apiLogger")

if (PUBLIC_KEY):
	try:
		int(PUBLIC_KEY, 16)
		logger.debug('Seed value for JWT encryption received, generating public key from the seed.')
		key = nacl.signing.SigningKey(PUBLIC_KEY, nacl.encoding.HexEncoder)
		ed25519_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes(key))
		pk = ed25519_key.public_bytes(encoding=Encoding.OpenSSH, format=PublicFormat.OpenSSH)
	except ValueError:
		logger.debug('Public key received.')
		pk = PUBLIC_KEY
else:
	logger.error('JWT public key not found. Please set a value of environment variable JWT_PUBLIC_KEY before running the application!')
	sys.exit(3)



def token_required(f):
	@wraps(f)
	def decorated(*args, **kwargs):
		try:
			auth = request.headers['Authorization'].split(maxsplit=2)
		except KeyError:
			logger.error('Authorization header is missing in the API request!')
			error = [{'code': 'invalid_token', 'status': '401', 'title': 'invalid_token', 'detail': 'Missing Authorization header'}]
			err = models.Error(many=True).dump(error)
			return J(err), 401
		else:
			try:
				if len(auth) > 1 and auth[0].lower() == 'bearer':
					logger.debug('JWT token received, 10 last symbols of the token are: {}'.format(auth[1][-10:]))
					token = auth[1]
					token_data = jwt.decode(token, pk, audience=AUDIENCE, options={
						'verify_exp': True, 
						'verify_aud': True,
						'verify_iss': False,
						'verify_iat': False
						})
					logger.debug('JWT validation succesful, 10 last symbols of the token are: {}'.format(token[-10:]))
					customer_id = token_data['customer_id']
					logger.debug('Customer ID = '.format(customer_id))
			except (jwt.ExpiredSignatureError) as e:
				logger.exception(e)
				error = [{'code': 'forbidden', 'status': '401', 'title': 'invalid_token', 'detail': "Invalid JWT token : Expired Signature (exp)"}]
				err = models.Error(many=True).dump(error)
				return J(err), 401
			except (jwt.DecodeError, jwt.exceptions.InvalidAudienceError, TypeError, KeyError) as e:
				logger.exception(e)
				error = [{'code': 'forbidden', 'status': '403', 'title': 'invalid_token', 'detail': "You aren't authorized to access this resource or take this action."}]
				err = models.Error(many=True).dump(error)
				return J(err), 403
			return f(*args, **kwargs)

	return decorated

			

@app.route('/' + AUDIENCE)
@token_required
def recommendation_list():
	try:
		filters = request.args.to_dict()
		validate_filters(filters)
		incomming_request_headers = get_headers()
		customer_id = jwt.decode(request.headers['Authorization'].split(maxsplit=2)[1], pk, audience=AUDIENCE)['customer_id']

		pool = ThreadPool(processes=5)
		async_history = pool.apply_async(get_order_history, (incomming_request_headers, customer_id,))
		async_customer = pool.apply_async(get_customer_properties, (filters.get('filter[meal_plan]'), incomming_request_headers, customer_id,))
		async_availabilities = pool.apply_async(get_available_recipes, (filters.get('filter[delivery_date]'), filters.get('filter[portion_count_per_meal]'), filters.get('filter[excluded_food_groups]'), incomming_request_headers,))
		async_history.wait()
		async_macros = pool.apply_async(get_order_history_macros, (async_history.get(), filters.get('filter[portion_count_per_meal]'),incomming_request_headers,))

		#print("########### CUSTOMER #########")
		async_customer.wait()
		c = async_customer.get()


		#print("########### AVAILABLE RECIPES #########")
		async_availabilities.wait()
		av_rec = async_availabilities.get()


		#print("########### ORDER HISTORY #########")
		async_macros.wait()
		h = async_macros.get()
		
		#print("########### RECOMMENDER INPUT #########")
		rec_features = models.Recommender_input(user=c, suggestions=av_rec, lastorder=h[0], lastorder2=h[1])

		#print("########### RECOMMENDER OUTPUT #########")
		recommendation = recommender.recommender(to_dict(rec_features), filters.get('filter[portion_count_per_meal]'))


		data = models.RecipeRecommendations(many=True).dump(recommendation)
		return J(data), 200

	except BadRequestException as e:
		logger.error(e.message)
		error = [{'code': 'bad_request', 'status': '400', 'title': 'Request Error', 'detail': e.message}]
		err = models.Error(many=True).dump(error)
		return J(err), 400

	except NotFoundException as e:
		logger.error(e.message)
		error = [{'code': 'not_found', 'status': '404', 'title': 'Not Found', 'detail': ''}]
		err = models.Error(many=True).dump(error)
		return J(err), 404

	except Exception as e:
		logger.exception(e)
		error = [{'code': 'internal_server_error', 'status': '500', 'title': 'Internal Server Error', 'detail': "We've notified our engineers and hope to address this issue shortly."}]
		err = models.Error(many=True).dump(error)
		return J(err), 500


if __name__ == '__main__':
	app.run(debug=False)