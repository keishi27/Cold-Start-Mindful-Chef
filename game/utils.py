from loguru import logger
import pandas as pd
import psycopg2  # import the python driver for PostgreSQL


def get_connection():
    logger.info("Handling connection to PostgreSQL database")

    connection = psycopg2.connect(
        user="s2ds2020",
        password="pb22b61c52805a66929959dc484393e286e63ef48397c930aa3a312723dc9abff",
        host="ec2-3-248-70-223.eu-west-1.compute.amazonaws.com",
        port="5432",
        database="d8s80rg6c5ojul",
    )
    return connection


def run_query(query):
    logger.info(f"Running query: {query}")

    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    return df


if __name__ == "__main__":
    query = """
        SELECT * 
        FROM deliveries 
        LIMIT 15
        """

    df = run_query(query=query)
    logger.info(df.head())
