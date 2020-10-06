from loguru import logger

from utils import run_query

query = """
        SELECT * 
        FROM deliveries 
        LIMIT 15
        """

df = run_query(query=query)
logger.info(df.head())
