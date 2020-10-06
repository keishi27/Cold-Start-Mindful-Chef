import pandas as pd
import numpy as np
from loguru import logger
import re

import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import (
    TfidfTransformer,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn import preprocessing, decomposition

from sklearn.cluster import AgglomerativeClustering, KMeans

import gensim
from gensim.utils import simple_preprocess
import spacy  # for lemmatization

from wordcloud import WordCloud

nltk.download("stopwords")
stop_words = stopwords.words("english")

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en", disable=["parser", "ner"])

additional_stopwords = [
    "serve",
    "sustainably",
    "sustainable",
    "catch",
    "alongside",
    "make",
    "making",
    "made",
    "dish",
    "organic",
    "dragonfly",
    "caught",
    "coat",
    "coated",
    "wilt",
    "top",
    "topping",
    "topped",
    "serve",
    "serving",
    "served",
    "baby",
    "healthy",
    "fresh",
    "freshly",
    "red",
    "drizzle",
    "drizzling",
    "drizzled",
    "add",
    "added",
    "adding",
    "perfectly",
    "perfect",
    "sit",
    "refresh",
    "refreshing",
    "additional",
    "enough",
    "feature",
    "features",
]
stop_words.extend(additional_stopwords)


def get_unique_values(df, column):
    logger.info("Getting unique values")
    df = df.copy()

    df_count = (
        df.groupby(by=[column])
        .agg({"id": "count"})
        .reset_index()
        .loc[:, [column, "id"]]
    )
    return df_count


def _remove_punctuation_and_symbols(text):
    # search for these symbols and replace with empty string
    text = re.sub(r"[!?@#$+%*:;()/'-.<>]", "", text)
    cleanr = re.compile("<.*?>")
    cleaned_text = re.sub(cleanr, "", text)
    return cleaned_text


def _remove_digits_and_stopwords(text):
    # lowercase text
    text = text.lower()

    # remove digits
    rem_num = re.sub("[0-9]+", "", text)

    # remove stopwords
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if w not in stop_words]

    return " ".join(filtered_words)


def preprocess_text_columns(df, columns):
    logger.info("Preprocessing text columns")
    df = df.copy()
    for col in columns:

        logger.info(f"Stripping punctuation and symbol for {col}")
        df[col] = df[col].astype(str).apply(_remove_punctuation_and_symbols)

        logger.info(f"Stripping digits and stopwords for {col}")
        df[col] = df[col].astype(str).apply(_remove_digits_and_stopwords)

    return df


def _count_allergens(x):
    if x == "":
        return 0
    else:
        x = x.replace("{", "").replace("}", "")
        x = x.split(",")
        return len(x)


def count_allergens(df, column):
    logger.info("Replacing allergens with count")
    df = df.copy()
    df[column] = df[column].replace(np.nan, "", regex=True)
    df[column] = df[column].apply(_count_allergens)
    return df


def _split_allergens(df, column, prefix):
    """Makes one colums per allergen and apply .cat.code to them.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        training data containing the col "allergens"
    column: str
        name of column to process
    prefix: str
        string to rename columns

    Returns
    -------
    df: pandas.core.frame.DataFrame
        containing one column per allergen
    """
    df = df.copy()
    df[column] = df[column].astype("str").map(lambda x: re.sub(r"\W+", " ", x))
    df = df.join(df[column].str.split(" ", expand=True).add_prefix(prefix).fillna(0))
    df = df.drop(column, axis=1)
    return df


def _cat_encode_columns(df, col_list):
    # cat.code is applied to allergens after splitting the pairs
    for column in col_list:
        df[column] = df[column].astype("category").cat.codes
    return df


def get_allergens_encoding(df):
    logger.info("Getting allergens encoding")
    df = df.copy()

    df = _split_allergens(df=df, column="allergens", prefix="allergen")
    allergens_list = [allergen for allergen in df.filter(regex="^allerge.*").columns]
    return _cat_encode_columns(df=df, col_list=allergens_list)


def drop_columns(df, columns):
    logger.info(f"Dropping columns: {columns}")
    df = df.copy()
    return df.drop(columns=columns, axis=1)


def fill_missing_values(df, column, name):
    logger.info(f"Filling nan values of column {column} with {name}")
    df = df.copy()
    df[column] = df[column].fillna(name)
    return df


def label_encoding(df, column):
    logger.info(f"Label encoding for column: {column}")
    df = df.copy()
    le = preprocessing.LabelEncoder()
    le.fit(df[column])
    logger.info(f"unique classes: {le.classes_}")
    df[column] = le.transform(df[column])
    return df


def hot_encoding(df, columns):
    logger.info(f"Hot encoding for columns: {columns}")
    df = df.copy()
    return pd.get_dummies(df, columns=columns)


def count_encoding(df, column):
    logger.info(f"Count encoding for column: {column}")
    df = df.copy()
    df[column] = df[column].astype("object").map(df[column].value_counts())
    return df


def apply_tfidf(X_text, max_features=20):
    """Apply TFIDF to dataframe with text columns.

    Parameters
    ----------
    X_text: pandas.core.frame.DataFrame
        training data containing text columns
    max_featurea: int
        maximum number of top words

    Returns
    -------
    X_text_tfidf_df: pandas.core.frame.DataFrame
        with columns given by top words and values given by tfidf
    """

    count_vect = CountVectorizer(max_features=max_features)
    tfidf_transformer = TfidfTransformer()
    X_text_counts = count_vect.fit_transform(X_text)
    X_text_tfidf = tfidf_transformer.fit_transform(X_text_counts)

    logger.info(X_text_tfidf.shape)

    X_text_tfidf_df = pd.DataFrame(
        X_text_tfidf.todense(), columns=count_vect.get_feature_names()
    )
    X_text_tfidf_df.columns = [str(col) + "_tfidf" for col in X_text_tfidf_df.columns]

    return X_text_tfidf_df


def sent_to_words(sentences):
    for sentence in sentences:
        yield (
            gensim.utils.simple_preprocess(str(sentence), deacc=True)
        )  # deacc=True removes punctuations


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


def get_statistics_clusters(X, columns):
    number_of_clusters = len(X.cluster.unique())

    for k in range(0, number_of_clusters):
        logger.info(f"cluster: {k}")
        df = X.loc[X.cluster == k, columns].describe()
        logger.info(f"\n {df}")


def get_distribution_numerical_variable(dataframe, cluster, column, origin="script"):

    df = dataframe.copy()

    plt.figure(figsize=(13, 5))
    sns.distplot(df[column], color="blue")
    plt.title(column)
    plt.subplots_adjust(hspace=0.3)
    plt.axvline(df[column].mean(), color="black", linestyle="dashed", label="mean")
    plt.axvline(df[column].std(), color="red", linestyle="dotted", label="std")
    plt.legend(loc="upper right")

    if origin=="script":
        plt.savefig('clustering_plots/%s_distribution_cluster%d.pdf' %(str(column), cluster))

    logger.info(f"Distribution of {column}")


def get_hist_per_feature(dataframe, cluster, by_feature, feature_to_count,  origin="script"):#, ticklabels=[]):
    df = dataframe.copy()
    plt.figure(figsize=(15, 6))
    df.groupby(by=by_feature)[feature_to_count].count().plot(kind="bar")
    #x = df.groupby(by=by_feature)[feature_to_count].count().plot(kind="bar")
    plt.legend(loc="upper right")

    #if len(ticklabels)>0: 
    #   x.set_xticklabels(ticklabels)

    if origin=="script":
        plt.savefig('clustering_plots/%s_hist_cluster%d.pdf' %(str(feature_to_count), cluster))

    logger.info(df.groupby(by=by_feature)[feature_to_count].count())


    def get_top_words(df, n_comp, n_words):
    """get the most "important" words for a given categorical feature.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe to process
    n_components: int
        number of components used by TruncatedSVD class        
    n_words: int
        number of words to display

    Returns
    -------
    np.array: array with most important words sorted in a descending order
    """
    vectorizer = TfidfVectorizer(stop_words="english", strip_accents="unicode")
    vec = vectorizer.fit_transform(df)
    tfidf = pd.DataFrame(vec.todense(), columns=vectorizer.get_feature_names())

    svd = TruncatedSVD(n_components=n_comp, n_iter=7, random_state=42)
    svd_ft = svd.fit_transform(tfidf)

    df_comp = pd.DataFrame(svd.components_, columns=tfidf.columns)
    df_comp = df_comp.transpose()

    return np.mean(df_comp.abs(), axis=1).sort_values(ascending=False).head(n_words)


def show_words_cloud(df, column, n_components, n_words):

    """plot the word cloud for a given categorical feature.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe to process
    column: str
        name of column to process
    n_components: int
        number of components used by get_top_words function        
    n_words: int
        number of words to display

    Returns
    -------
    plt.figure: wordcloud for every cluster
    """
    for cluster_number in df["cluster"].unique():

        titles = get_top_words(
            df[df["cluster"] == cluster_number][column], n_components, n_words
        )
        df_titles = pd.DataFrame(titles)
        title_lists = df_titles.index.values

        plt.figure(figsize=(10, 8))
        wordcloud = WordCloud(
            background_color="white", mode="RGB", width=2000, height=1000
        ).generate(" ".join(title_lists))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()


# Features scaling and PCA
def feat_scaler(scaler, df):
    """get feature scaling.

    Parameters
    ----------
    scaler: sklearn.preproceesser
        type of scaling
    array: np.array
        data to process        
        
    Returns
    -------
    np.array: array with scaled features
    """
    data_scaled = scaler.fit_transform(df)
    return data_scaled


def pca_(df, n_components):
    """get principal components from data frame.

    Parameters
    ----------
    choice: string
        food group
    n_components: int
        number of components used by TruncatedSVD class        
        
    Returns
    -------
    np.array: array with principal components
    """
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(df)
    X = pca.transform(df)
    return X


def get_statistics_clusters(X, columns):  # to fix
    number_of_clusters = len(X.cluster.unique())

    for k in range(0, number_of_clusters):
        logger.info(f"cluster: {k}")
        df = X.loc[X.cluster == k, columns].describe()
        logger.info(f"\n {df}")
        # return df


# clustering #draft, to fix Function used to investigate WCSS for clustering within food groups
def WCSS_plotter(X_all, X_v, X_m, X_f, X_mv, X_mf, X_fv):
    # Using the elbow method to find the optimal number of clusters
    k = 10
    wcss_v = []
    wcss_m = []
    wcss_f = []
    wcss_mv = []
    wcss_mf = []
    wcss_fv = []
    wcss = []

    for i in range(1, k):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(X_all)
        wcss.append(kmeans.inertia_)

    for i in range(1, k):
        kmeans_v = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans_v.fit(X_v)
        wcss_v.append(kmeans_v.inertia_)

    for i in range(1, k):
        kmeans_m = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans_m.fit(X_m)
        wcss_m.append(kmeans_m.inertia_)

    for i in range(1, k):
        kmeans_f = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans_f.fit(X_f)
        wcss_f.append(kmeans_f.inertia_)

    for i in range(1, k):
        kmeans_mv = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans_mv.fit(X_mv)
        wcss_mv.append(kmeans_mv.inertia_)

    for i in range(1, k):
        kmeans_mf = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans_mf.fit(X_mf)
        wcss_mf.append(kmeans_mf.inertia_)

    for i in range(1, k):
        kmeans_fv = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans_fv.fit(X_fv)
        wcss_fv.append(kmeans_fv.inertia_)

    plt.plot(range(1, 10), wcss, label="all")
    plt.plot(range(1, 10), wcss_v, label="vegan")
    plt.plot(range(1, 10), wcss_m, label="meat")
    plt.plot(range(1, 10), wcss_f, label="fish")
    plt.plot(range(1, 10), wcss_mv, label="meat and vegan")
    plt.plot(range(1, 10), wcss_mf, label="meat and fish")
    plt.plot(range(1, 10), wcss_fv, label="fish and vegan")

    plt.legend()

    plt.title("The Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")

    plt.show()



