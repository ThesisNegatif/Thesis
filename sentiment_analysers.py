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
nltk.download('wordnet')
from nltk import tokenize
from justification_miners import JustificationMiner
from classifier_trainers import TrainClassifier

class SentimentAnalyser(object):
    def get_mined_sentiment(self, corpus, vocab, model, clustering_model=['Kmeans','Agglomerative','DBSCAN','MeanShift'], numericiser=['BoW','SBERT'], embedder=['Wiki', 'NLI'], aggregation=['Mean','Mode','MeanRounded']):
        """
        Args:
            corpus(list): List of sentences on which to predict sentiment
            vocab(dict): Mapping dictionary created from training dataset
            model(model): Trained classifier model
            clustering_model(str): Clustering model to use. Options are 'Kmeans', 'Agglomerative', 'DBSCAN', 'MeanShift'
            numericiser(str): Numericisation method to use. Options are Bag of Words ('BoW') and 'SBERT'
            embedder(str): Pretrained dataset to use for SBERT. Options are Wikipedia ('Wiki') and 'NLI'
            aggregation(str): Sentiment aggregation method to use. Options are unrounded Mean scores ('Mean'), 'Mode', and rounded Mean scores ('MeanRounded')
        Returns:
            justification_score(int): Aggregated summarised sentiment score predicted by model
        """
        J = JustificationMiner()
        C = TrainClassifier()

        justifications = J.mine_justifications(corpus, clustering_model=clustering_model, num_clusters=5, save_data=False, embedder=embedder)

        if numericiser=='BoW':
            X_test = C.create_X_test(justifications, vocab)
            justifications_predictions = model.predict(X_test)
        elif numericiser=='SBERT':
            X_test = C.create_X_test_embeddings(justifications, embedder=embedder)
            justifications_predictions = model.predict(X_test)

        if aggregation=='Mean':
            justification_score = (justifications_predictions).mean()
        elif aggregation=='Mode':
            justification_score = stats.mode(justifications_predictions)[0][0]
        elif aggregation=='MeanRounded':
            justification_score = (justifications_predictions).mean().round()

        return justification_score

    def get_full_sentiment(self, corpus, vocab, model, numericiser=['BoW','SBERT'], embedder=['Wiki','NLI'], aggregation=['Mean','Mode','MeanRounded']):
        """
        Args:
            corpus(list): List of sentences on which to predict sentiment
            vocab(dict): Mapping dictionary created from training dataset
            model(model): Trained classifier model
            numericiser(str): Numericisation method to use. Options are Bag of Words ('BoW') and 'SBERT'
            embedder(str): Pretrained dataset to use for SBERT. Options are Wikipedia ('Wiki') and 'NLI'
            aggregation(str): Sentiment aggregation method to use. Options are unrounded Mean scores ('Mean'), 'Mode', and rounded Mean scores ('MeanRounded')
        Returns:
            full_score(int): Aggregated full sentiment score predicted by model
        """
        C = TrainClassifier()

        sentences = tokenize.sent_tokenize(corpus)

        if numericiser=='BoW':
            X_test = C.create_X_test(sentences, vocab)
            predictions = model.predict(X_test)
        elif numericiser=='SBERT':
            X_test = C.create_X_test_embeddings(sentences, embedder=embedder)
            predictions = model.predict(X_test)

        if aggregation=='Mean':
            full_score = (predictions).mean()
        elif aggregation=='Mode':
            full_score = stats.mode(predictions)[0][0]
        elif aggregation=='MeanRounded':
            full_score = (predictions).mean().round()

        return full_score
