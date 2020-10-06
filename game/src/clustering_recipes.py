import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from pprint import pprint
import re
import shutil
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim 

from nltk.corpus import stopwords

from game import clustering_utils

data = pd.read_csv('../data/recipe_table.csv', sep=';')

data = clustering_utils.drop_columns(df=data, columns=[
    "specials", 
    "image_url",
    "instructions"
    ])

#Fill nan values
data = clustering_utils.fill_missing_values(df=data, column= "food_group", name='other')
data = clustering_utils.fill_missing_values(df=data, column= "cuisine", name='other')
data = clustering_utils.fill_missing_values(df=data, column="season", name='other')

#Encoding allergens
data = clustering_utils.count_allergens(df=data, column="allergens")
#Label encoding: cuisine
data = clustering_utils.label_encoding(df=data, column="cuisine")
#Hot encoding: season
data = clustering_utils.hot_encoding(df=data, columns=["season"])

#Creating food group dictonary
food_mapping = clustering_utils.get_unique_values(df=data, column="food_group")
food_mapping.reset_index
dict_food_group = food_mapping.set_index('id')['food_group'].to_dict()

#Count encoding: food group
data = clustering_utils.count_encoding(df=data, column="food_group")

######### Apply topic modelling to: title, key_ingredient, description #########

data['text_union'] = (data["key_ingredient"]+str(' ')
                      +data["title"]+str(' ')
                      +data["description"])

X_text = data['text_union']
data_words = list(clustering_utils.sent_to_words(X_text))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove Stop Words
data_words_nostops = clustering_utils.remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = clustering_utils.make_bigrams(data_words_nostops, bigram_mod)
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = clustering_utils.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Compute Perplexity: a measure of how good the model is (the lower, the better)
perplexity = lda_model.log_perplexity(corpus)
print('\nPerplexity: ', perplexity) 
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

array_topics = []
# Get main topic in each document
for i, row in enumerate(lda_model[corpus]):
    topic_importance = sorted(row[0], key=lambda x: (x[1]), reverse=True)
    array_topics.append(topic_importance[0][0])

data["topic"] = array_topics

data = clustering_utils.drop_columns(df=data, columns=[
    "key_ingredient", 
    "title",
    "description",
    "text_union"
    ])


######### Clustering ########

X = data.drop(columns=["id"], axis=1).copy()
# Using the elbow method to find the optimal number of clusters
k = 10
distortions = []
for i in range(1, k):
    kmean_model = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmean_model.fit(X)
    distortions.append(sum(np.min(cdist(X, kmean_model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
plt.plot(range(1, k), distortions)
plt.title('Elbow plot')
plt.xlabel('Number of clusters')
plt.ylabel('Distortions')
plt.savefig('clustering_plots/elbow_plot.pdf')

#According to the Elbow graph we deterrmine the clusters number as 4. 
#Applying k-means algorithm to the X dataset.
kmeans = KMeans(n_clusters=4, init ='k-means++', random_state=42)

y_kmeans = kmeans.fit_predict(X)

#PCA
X_pca = X.copy()
pca = decomposition.PCA(n_components  = 2)
pca.fit(X_pca)
X_pca = pca.transform(X_pca)

# Visualising the clusters
x = X_pca[0]
y = X_pca[1]

plt.scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1])
plt.scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1])          
plt.scatter(X_pca[y_kmeans == 2, 0], X_pca[y_kmeans == 2, 1])
plt.scatter(X_pca[y_kmeans == 3, 0], X_pca[y_kmeans == 3, 1])

#Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
#plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 8], s=200, c='yellow', label = 'Centroids')
plt.title('Clusters of Recipes')
plt.xlabel("X_pca1")
plt.ylabel("X_pca2")
plt.savefig('clustering_plots/clusters.pdf')

X["cluster"] = y_kmeans

# Statistical analysis of clusters
clustering_utils.get_statistics_clusters(X, columns=["carbs", "calories", "protein", "fat", "topic"]) 

X['food_group'] = data.food_group
number_of_clusters = 4

dfs = []
for k in range(0, number_of_clusters):
    df = X.loc[X.cluster==k, :]
    logger.info(f"elements in cluster {k}: {len(df)}")
    dfs.append(X.loc[X.cluster==k, :])

# Topic statistics
clustering_utils.get_distribution_numerical_variable(dataframe=dfs[0], cluster=0, column="topic", origin="script")
clustering_utils.get_hist_per_feature(dataframe=dfs[0], cluster=0, by_feature="topic", feature_to_count="topic", origin="script")

clustering_utils.get_distribution_numerical_variable(dataframe=dfs[1], cluster=1, column="topic", origin="script")
clustering_utils.get_hist_per_feature(dataframe=dfs[1], cluster=1, by_feature="topic", feature_to_count="topic", origin="script")

clustering_utils.get_distribution_numerical_variable(dataframe=dfs[2], cluster=2, column="topic", origin="script")
clustering_utils.get_hist_per_feature(dataframe=dfs[2], cluster=2, by_feature="topic", feature_to_count="topic", origin="script")

clustering_utils.get_distribution_numerical_variable(dataframe=dfs[3], cluster=3, column="topic", origin="script")
clustering_utils.get_hist_per_feature(dataframe=dfs[3], cluster=3, by_feature="topic", feature_to_count="topic", origin="script")

#Food group statistics

clustering_utils.get_distribution_numerical_variable(dataframe=dfs[0], cluster=0, column="food_group", origin="script")
clustering_utils.get_hist_per_feature(dataframe=dfs[0].replace({"food_group": dict_food_group}), 
                                     cluster=0, by_feature="food_group", feature_to_count="food_group", origin="script")

clustering_utils.get_distribution_numerical_variable(dataframe=dfs[1], cluster=1, column="food_group", origin="script")
clustering_utils.get_hist_per_feature(dataframe=dfs[1].replace({"food_group": dict_food_group}), 
                                     cluster=1, by_feature="food_group", feature_to_count="food_group", origin="script")

clustering_utils.get_distribution_numerical_variable(dataframe=dfs[2], cluster=2, column="food_group", origin="script")
clustering_utils.get_hist_per_feature(dataframe=dfs[2].replace({"food_group": dict_food_group}), 
                                     cluster=2, by_feature="food_group", feature_to_count="food_group", origin="script")

clustering_utils.get_distribution_numerical_variable(dataframe=dfs[3], cluster=3, column="food_group", origin="script")
clustering_utils.get_hist_per_feature(dataframe=dfs[3].replace({"food_group": dict_food_group}), 
                                     cluster=3, by_feature="food_group", feature_to_count="food_group", origin="script")

#Data frame for game: recipe_ids per cluster
X['id'] = data.id
X_game = X.groupby(['cluster'])["id"].agg(['count']).rename(columns={"count":"recipe_id_count"}).reset_index()
X_game = X.loc[:,['cluster', 'id']]
X_game.sort_values("cluster", axis = 0, ascending = True, inplace = True)
logger.info(X_game)
X_game.to_csv(r'df_game.csv', index = False)

shutil.copy('df_game.csv', '../data/')
os.remove('df_game.csv')