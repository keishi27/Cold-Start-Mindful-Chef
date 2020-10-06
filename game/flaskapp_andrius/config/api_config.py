#Recipe recommender API configuration file

import os

#Public key for validating JWT tokens in incoming API requests
PUBLIC_KEY = os.environ.get('JWT_PUBLIC_KEY')

#API endpoint for getting recipe recommendations
AUDIENCE = 'recipes/recommendations'
 #'/recipes/recommendations'#'/

#MC data API endpoints for collecting data required for producing recomendation
DATA_API_URL = os.environ.get('DATA_API_URL', 'https://api.staging.mindfulchef.com')
RECIPE_AVAILABILITY_ENDPOINT = os.environ.get('RECIPE_AVAILABILITY_ENDPOINT', '/recipes/availabilities')
RECIPE_MACROS_ENDPOINT = os.environ.get('RECIPE_MACROS_ENDPOINT', '/recipes/recipes')
CUSTOMER_ENDPOINT = os.environ.get('CUSTOMER_ENDPOINT', '/customers/customers')
SUBSCRIPTIONS_ENDPOINT = os.environ.get('SUBSCRIPTIONS_ENDPOINT', '/subscriptions/subscriptions')
DELIVERIES_ENDPOINT = os.environ.get('DELIVERIES_ENDPOINT', '/deliveries/deliveries') 