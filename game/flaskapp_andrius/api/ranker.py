import pickle
import pandas as pd
import numpy as np
import os

from sys import stdout
import logging

logger = logging.getLogger("apiLogger")


clf = ""
try:
    with open(os.path.dirname(os.path.realpath(__file__)) + "/pretrained/classifier.p", "rb") as f:
        clf = pickle.load(f)
except IOError:
    logger.exception("classifier.p file not found!")
    raise

def rankrecipes(df, clf = clf):

    '''

    Takes a preprocessed query as a dataframe and returns the ranks for each suggestion.

    (list of np ints) <- df, model

    '''
    logger.debug("Ranking suggestions \n {}".format(df.to_string()))
    
    probs = [p[1] for p in clf.predict_proba(df)]
    ranks = len(probs) - np.argsort(probs)
    logger.debug("Ranks genereted for the suggestions: {}".format(ranks))
    return(ranks)

