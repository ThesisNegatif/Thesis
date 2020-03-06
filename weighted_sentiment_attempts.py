from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import json
import nltk
import os
import random
import re
from statistics import mean
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import Counter
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from tqdm import tqdm
import collections
from scipy import stats


# DONE/To-do: Move read txt function out into a separate function from justification mining
# Code to use this in python env is import justification_miners as jm // jm.JustificationMiner(arguments)
# To-do: Want a way to retrieve cluster_centres, clusters, dataframe.head, justifications


# def read_file():
#     """
#     Args:
#         None
#     Returns:
#         corpus(str): Full text corpus read in as a string.
#     """
#     overview_file = input("Enter txt file name of data: ")
#     with open(overview_file, 'r') as file:
#         initial_corpus = file.read()
#     corpus = initial_corpus.split('. ')
#     return corpus
# # file to be read in line 29 is: 'JWN_Nordstrom_MDNA_overview_2017.txt'
# corpustwo = read_file()

def read_file():
    """
    Args:
        None
    Returns:
        corpus(str): Full text corpus read in as a string.
    """
    #overview_file = input("Enter txt file name of data: ")
    with open('JWN_Nordstrom_MDNA_overview_2017.txt', 'r') as file:
        initial_corpus = file.read()
    corpus = initial_corpus.split('. ')
    return corpus
# file to be read in line 29 is: 'JWN_Nordstrom_MDNA_overview_2017.txt'
corpustwo = read_file()


# Function to be made into a class is below.
def JustificationMiner(corpus, clustering_model=['Kmeans','Agglomerative'], num_clusters=5, save_data=False):
    """
    Args:
        corpus(str): Text corpus on which to perform Justification Mining.
        clustering_model(str): Clustering model to use. Options are 'Kmeans' and 'Agglomerative'.
        num_clusters(int): Number of clusters returned by clustering models. Defaults to 5.
        save_data(bool): Whether to save Pandas dataframes of both the full data and max cosines
            similarity scores data. Saves in two new csv files within same directory. Defaults to False.
    Returns:
        justifications(list): Full list of justification sentences returned by the miner.
    """
    # Extract sentence embeddings
    embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
    corpus_embeddings = embedder.encode(corpus)

    # Perform KMeans or Agglomerative clustering
    # num_clusters = 5
    if clustering_model=='Kmeans':
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        # labels = clustering_model.predict(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        cluster_centres = clustering_model.cluster_centers_
    elif clustering_model=='Agglomerative':
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        clustered_embeddings = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_embeddings[cluster_id].append(corpus_embeddings[sentence_id])
        cluster_centres = [np.mean(cluster, axis=0) for cluster in clustered_embeddings]

    # Create DataFrame with all data
    cluster_centres_list = [cluster_centres[label] for label in cluster_assignment]
    # cluster_centres_list = []
    # for label in cluster_assignment:
    #     cluster_centres_list.append(cluster_centres[label])
    all_data_df = pd.DataFrame()
    all_data_df['sentences'] = corpus
    all_data_df['embeddings'] = corpus_embeddings
    all_data_df['labels'] = cluster_assignment
    all_data_df['cluster_centres'] = cluster_centres_list

    cosine_similarities_list = [cosine_similarity(x.reshape(1,-1),y.reshape(1,-1)) for x, y in zip(all_data_df['embeddings'], all_data_df['cluster_centres'])]
    cosine_similarities_list_final = [cosines[0][0] for cosines in cosine_similarities_list]
    all_data_df['cosine_similarities_to_centre'] = cosine_similarities_list_final
    new_df = all_data_df.sort_values(['labels', 'cosine_similarities_to_centre'], ascending=False).drop_duplicates(['labels'])
    justifications = new_df['sentences'].tolist()
    # print("Justifications:")
    # print(justifications)
    # print("")

    if save_data==True:
        all_data_df.to_csv('all_data_df.csv')
        new_df.to_csv('new_df.csv')

    # Print Clusters
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        #print(cluster_id)
    #print(clustered_sentences)
    # for i, cluster in enumerate(clustered_sentences):
    #     print("Cluster ", i+1)
    #     print(cluster)
    #     print("")

    return justifications

justificationstwo = JustificationMiner(corpustwo, clustering_model='Agglomerative', num_clusters=5, save_data=False)

# Train classifier
data_csv = pd.read_csv(filepath_or_buffer='Sentences_75Agree_csv.csv' , sep='.@', header=None, names=['sentence','sentiment'], engine='python')

list_data = []
for index, row in data_csv.iterrows():
    dictionary_data = {}
    dictionary_data['message_body'] = row['sentence']
    if row['sentiment'] == 'positive':
         dictionary_data['sentiment'] = 2
    elif row['sentiment'] == 'negative':
         dictionary_data['sentiment'] = 0
    else:
         dictionary_data['sentiment'] = 1
    #dictionary_data['sentiment'] = row['sentiment']
    list_data.append(dictionary_data)

dictionary_data = {}
dictionary_data['data'] = list_data

messages = [sentence['message_body'] for sentence in dictionary_data['data']]
sentiments = [sentence['sentiment'] for sentence in dictionary_data['data']]
nltk.download('wordnet')

def preprocess(message):
    """
    # Parameters
    # ----------
    #     message : The text message to be preprocessed.
    #
    # Returns
    # -------
    #     tokens: The preprocessed text into tokens.
    """
    text = message.lower()
    text = re.sub(r'[^\w\s]|_', ' ', text)
    tokens = text.split()
    wnl = nltk.stem.WordNetLemmatizer()
    tokens = [wnl.lemmatize(token) for token in tokens if len(token)>1]
    return tokens

tokenized = [preprocess(message) for message in messages]
bow = Counter([j for i in tokenized for j in i])
freqs = {key: value/len(tokenized) for key, value in bow.items()}
low_cutoff = 0.00029
high_cutoff = 5
K_most_common = [x[0] for x in bow.most_common(high_cutoff)]
filtered_words = [word for word in freqs if word not in K_most_common]
vocab = {word: i for i, word in enumerate(filtered_words, 1)}
id2vocab = {i: word for word, i in vocab.items()}
filtered = [[word for word in message if word in vocab] for message in tokenized]
# This part might need to be redone as I didn't use the balancing function
token_ids = [[vocab[word] for word in message] for message in filtered]

X_train = token_ids
y_train = sentiments
# Average for training data length is 16
# avg_length = []
# for sentence in X_train:
#     avg_length.append(len(sentence))
# print(len(avg_length))
# print('average is:', mean(avg_length))
# There's 450 sentences longer than len 25 and 170 longer than 30
# long_length = []
# for sentence in X_train:
#     if len(sentence) > 30:
#         long_length.append(len(sentence))
# print(len(long_length))

for i, sentence in enumerate(X_train):
    if len(sentence) <=30:
        X_train[i] = ((30-len(sentence)) * [0] + sentence)
    elif len(sentence) > 30:
        X_train[i] = sentence[:30]

# Random Forest
random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, y_train)

# model_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, min_samples_split=2)
# model_dt.fit(X_train, y_train)
#
# naive_bayes = MultinomialNB()
# naive_bayes.fit(X_train, y_train)
#
# SVM = SVC()
# SVM.fit(X_train, y_train)


tokenized_j = [preprocess(message) for message in justificationstwo]
bow_j = Counter([j for i in tokenized_j for j in i])
freqs_j = {key: value/len(tokenized_j) for key, value in bow_j.items()}
low_cutoff_j = 0.00029
high_cutoff_j = 5
K_most_common_j = [x[0] for x in bow_j.most_common(high_cutoff_j)]
filtered_words_j = [word for word in freqs_j if word not in K_most_common_j]
vocab_j = {word: i for i, word in enumerate(filtered_words_j, 1)}
id2vocab_j = {i: word for word, i in vocab_j.items()}
filtered_j = [[word for word in message if word in vocab_j] for message in tokenized_j]
# This part might need to be redone as I didn't use the balancing function
token_ids_j = [[vocab_j[word] for word in message] for message in filtered_j]

X_test = token_ids_j
for i, sentence in enumerate(X_test):
    if len(sentence) <=30:
        X_test[i] = ((30-len(sentence)) * [0] + sentence)
    elif len(sentence) > 30:
        X_test[i] = sentence[:30]

justifications_predictions_rf = random_forest.predict(X_test)
print(justificationstwo)
print(justifications_predictions_rf)
# Weighted calculation below: Note, this would be the unweighted sentiment to correlate to stock returns
print((justifications_predictions_rf/2).mean())
# Unweighted calculated using mode instead of mean:
print(stats.mode(justifications_predictions_rf/2)[0][0])

# justifications_predictions_dt = model_dt.predict(X_test)
# print(justifications_predictions_dt)
#
# justifications_predictions_nb = naive_bayes.predict(X_test)
# print(justifications_predictions_nb)
#
# justifications_predictions_svm = SVM.predict(X_test)
# print(justifications_predictions_svm)
