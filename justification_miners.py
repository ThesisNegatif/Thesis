from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, DBSCAN, MeanShift
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
import pickle
nltk.download('wordnet') #consider commenting this out because it might already be downloaded
from nltk import tokenize

class JustificationMiner(object):
    def mine_justifications(self, string_text, clustering_model=['Kmeans','Agglomerative','DBSCAN','MeanShift'], num_clusters=5, save_data=False, embedder=['Wiki','NLI']):
        """
        Args:
            string_text(str): Text on which to perform Justification Mining.
            clustering_model(str): Clustering model to use. Options are 'Kmeans', 'Agglomerative', 'DBSCAN', 'MeanShift'.
            num_clusters(int): Number of clusters returned by Agglomerative and KMeans clustering models. Defaults to 5.
            save_data(bool): Whether to save Pandas dataframes of both the full data and max cosines
                similarity scores data. Saves in two new csv files within same directory. Defaults to False.
            embedder(str): Dataset used by SBERT to encode sentences. Options are Wikipedia ('Wiki') and 'NLI'
        Returns:
            justifications(list): Full list of justification sentences returned by the miner.
        """
        # Split Sentences
        corpus = tokenize.sent_tokenize(string_text)

        # Extract sentence embeddings
        if embedder=='Wiki':
            embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
        elif embedder=='NLI':
            embedder = SentenceTransformer('bert-base-nli-mean-tokens')
        corpus_embeddings = embedder.encode(corpus)

        # Perform KMeans or Agglomerative clustering
        # num_clusters = 5
        if clustering_model=='Kmeans':
            clustering_model = KMeans(n_clusters=num_clusters)
            clustering_model.fit(corpus_embeddings)
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
        elif clustering_model=='MeanShift':
            clustering_model = MeanShift()
            clustering_model.fit(corpus_embeddings)
            cluster_assignment = clustering_model.labels_
            cluster_centres = clustering_model.cluster_centers_
        elif clustering_model=='DBSCAN':
            clustering_model = DBSCAN(eps=0.5)
            clustering_model.fit(corpus_embeddings)
            cluster_assignment = clustering_model.labels_
            number_of_clusters = len(set(cluster_assignment))
            clustered_embeddings = [[] for i in range(number_of_clusters)]
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                clustered_embeddings[cluster_id].append(corpus_embeddings[sentence_id])
            cluster_centres = [np.mean(cluster, axis=0) for cluster in clustered_embeddings]

        # Create DataFrame with all data
        cluster_centres_list = [cluster_centres[label] for label in cluster_assignment]
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

        if save_data==True:
            all_data_df.to_csv('all_data_df.csv')
            new_df.to_csv('new_df.csv')

        return justifications
