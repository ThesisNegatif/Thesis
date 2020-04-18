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

class TrainClassifier(object):
    def preprocess(self, message):
        """
        Args:
            message(str): The text message to be preprocessed
        Returns:
            tokens(list): The preprocessed text into tokens
        """
        text = message.lower()
        text = re.sub(r'[^\w\s]|_', ' ', text)
        tokens = text.split()
        wnl = nltk.stem.WordNetLemmatizer()
        tokens = [wnl.lemmatize(token) for token in tokens if len(token)>1]
        return tokens

    def create_X_train(self):
        """
        Args:
            None
        Returns:
            vocab(dict): The dictionary that maps words to tokens for BoW method
            X_train(list): The Malo et al., 2013 dataset tokenised with BoW and padded/truncated
            sentiments(list): The corresponding sentiment scores for X_train (from Malo et al., 2013)
        """
        data_csv = pd.read_csv(filepath_or_buffer='Sentences_75Agree_csv.csv' , sep='.@', header=None, names=['sentence','sentiment'], engine='python')
        list_data = []
        for index, row in data_csv.iterrows():
            dictionary_data = {}
            dictionary_data['message_body'] = row['sentence']
            if row['sentiment'] == 'positive':
                 dictionary_data['sentiment'] = 1
            elif row['sentiment'] == 'negative':
                 dictionary_data['sentiment'] = -1
            else:
                 dictionary_data['sentiment'] = 0 # For neutral sentiment
            list_data.append(dictionary_data)
        dictionary_data = {}
        dictionary_data['data'] = list_data
        messages = [sentence['message_body'] for sentence in dictionary_data['data']]
        sentiments = [sentence['sentiment'] for sentence in dictionary_data['data']]
        tokenized = [self.preprocess(message) for message in messages]

        # # Balancing training data due to large number of neutral sentences
        # balanced = {'messages': [], 'sentiments':[]}
        # n_neutral = sum(1 for each in sentiments if each == 1)
        # N_examples = len(sentiments)
        # # print(n_neutral/N_examples)
        # keep_prob = (N_examples - n_neutral)/2/n_neutral
        # # print(keep_prob)
        # for idx, sentiment in enumerate(sentiments):
        #     message = tokenized_unbalanced[idx]
        #     if len(message) == 0:
        #         # skip this sentence because it has length zero
        #         continue
        #     elif sentiment != 1 or random.random() < keep_prob:
        #         balanced['messages'].append(message)
        #         balanced['sentiments'].append(sentiment)
        # tokenized = balanced['messages']
        # sentiments_balanced = balanced['sentiments']

        bow = Counter([j for i in tokenized for j in i])
        # For X_test creation, remove the bow=Counter statement above and instead filter tokenized_test then continue with functions below.
        freqs = {key: value/len(tokenized) for key, value in bow.items()} #keys are the words in the vocab, values are the count of those words
        low_cutoff = 0.00029
        high_cutoff = 5
        K_most_common = [x[0] for x in bow.most_common(high_cutoff)] #most_common() is a method in collections.Counter
        filtered_words = [word for word in freqs if word not in K_most_common]
        vocab = {word: i for i, word in enumerate(filtered_words, 1)}
        id2vocab = {i: word for word, i in vocab.items()}
        filtered = [[word for word in message if word in vocab] for message in tokenized] # Here vocab is referring to vocab.keys()
        token_ids = [[vocab[word] for word in message] for message in filtered]

        # sentiments_balanced = balanced['sentiments']
        # # Unit test for balancing:
        # unique, counts = np.unique(sentiments_balanced, return_counts=True)
        # print(np.asarray((unique, counts)).T)
        # print(np.mean(sentiments_balanced))
        # ##################

        # Left padding and truncating to the same length
        X_train = token_ids
        for i, sentence in enumerate(X_train):
            if len(sentence) <=30:
                X_train[i] = ((30-len(sentence)) * [0] + sentence)
            elif len(sentence) > 30:
                X_train[i] = sentence[:30]

        return vocab, X_train, sentiments

    def create_X_test(self, sentences, vocab):
        """
        Args:
            sentences(list): The test sentences on which to predict
            vocab(dict): Mapping dictionary from training data
        Returns:
            X_test(list): The testing dataset tokenised with BoW and padded/truncated
        """
        tokenized = [self.preprocess(sentence) for sentence in sentences]
        filtered = [[word for word in sentence if word in vocab.keys()] for sentence in tokenized] # X_test filtered to only words in training vocab
        # Alternate method with functional programming:
        # filtered = [list(filter(lambda a: a in vocab.keys(), sentence)) for sentence in tokenized]
        token_ids = [[vocab[word] for word in sentence] for sentence in filtered] # Numericise data

        # Remove short sentences in X_test
        token_ids_filtered = [sentence for sentence in token_ids if len(sentence)>5]
        X_test = token_ids_filtered
        # print('X_test:', X_test)
        for i, sentence in enumerate(X_test):
            if len(sentence) <=30:
                X_test[i] = ((30-len(sentence)) * [0] + sentence)
            elif len(sentence) > 30:
                X_test[i] = sentence[:30]

        return X_test

    def train_classifier(self, classifier_model=['Decision_Tree','Random_Forest', 'Naive_Bayes', 'SVM']):
        """
        Args:
            classifier_model(str): Classifier used for training. Options are 'Decision_Tree', 'Random_Forest', 'Naive_Bayes' and 'SVM'
        Returns:
            model(model): The trained model
            vocab(dict): Mapping dictionary created from training data
        """
        vocab, X_train, sentiments = self.create_X_train()
        y_train = sentiments
        if classifier_model=='Decision_Tree':
            model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, min_samples_split=2, class_weight="balanced")
            model.fit(X_train, y_train)
        elif classifier_model=='Random_Forest':
            model = RandomForestClassifier(n_estimators=200, class_weight="balanced")
            model.fit(X_train, y_train)
        elif classifier_model=='Naive_Bayes':
            model = MultinomialNB()
            model.fit(X_train, y_train)
        elif classifier_model=='SVM':
            model = SVC(class_weight='balanced')
            model.fit(X_train, y_train)
        return model, vocab

    def create_X_train_embeddings(self, embedder=['Wiki','NLI']):
        """
        Args:
            embedder(str): Pretrained dataset to use for SBERT. Options are Wikipedia ('Wiki') and 'NLI'
        Returns:
            X_train_embeddings(list): Training data (from Malo et al., 2013) encoded into embeddings vectors with SBERT
            sentiment_embeddings(list): Corresponding sentiment scores for training data
        """
        data_csv = pd.read_csv(filepath_or_buffer='Sentences_75Agree_csv.csv' , sep='.@', header=None, names=['sentence','sentiment'], engine='python')
        list_data = []
        for index, row in data_csv.iterrows():
            dictionary_data = {}
            dictionary_data['message_body'] = row['sentence']
            if row['sentiment'] == 'positive':
                 dictionary_data['sentiment'] = 1
            elif row['sentiment'] == 'negative':
                 dictionary_data['sentiment'] = -1
            else:
                 dictionary_data['sentiment'] = 0 # For neutral sentiment
            list_data.append(dictionary_data)
        dictionary_data = {}
        dictionary_data['data'] = list_data
        messages = [sentence['message_body'] for sentence in dictionary_data['data']]
        sentiments_embeddings = [sentence['sentiment'] for sentence in dictionary_data['data']]

        # Extract sentence embeddings
        if embedder=='Wiki':
            embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
        elif embedder=='NLI':
            embedder = SentenceTransformer('bert-base-nli-mean-tokens')
        corpus_embeddings = embedder.encode(messages)
        X_train_embeddings = corpus_embeddings

        return X_train_embeddings, sentiments_embeddings

    def create_X_test_embeddings(self, sentences, embedder=['Wiki','NLI']):
        """
        Args:
            sentences(list): The test sentences on which to predict
            embedder(str): Pretrained dataset to use for SBERT. Options are Wikipedia ('Wiki') and 'NLI'
        Returns:
            X_test_embeddings(list): The testing dataset encoded into embeddings vectors with SBERT
        """
        # Extract sentence embeddings
        if embedder=='Wiki':
            embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
        elif embedder=='NLI':
            embedder = SentenceTransformer('bert-base-nli-mean-tokens')
        corpus_embeddings = embedder.encode(sentences)
        X_test_embeddings = corpus_embeddings
        return X_test_embeddings

    def train_classifier_embeddings(self, classifier_model=['Decision_Tree','Random_Forest', 'Naive_Bayes', 'SVM'], balanced=True, embedder=['Wiki','NLI']):
        """
        Args:
            classifier_model(str): Classifier used for training. Options are 'Decision_Tree', 'Random_Forest', 'Naive_Bayes' and 'SVM'
            balanced(bool): Whether to use class_weight parameter to balance training data labels. Defaults to True
            embedder(str): Pretrained dataset to use for SBERT. Options are Wikipedia ('Wiki') and 'NLI'
        Returns:
            model(model): The trained model
        """
        X_train, sentiments = self.create_X_train_embeddings(embedder=embedder)
        y_train = sentiments

        if balanced==True:
            if classifier_model=='Decision_Tree':
                model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, min_samples_split=2, class_weight="balanced")
                model.fit(X_train, y_train)
            elif classifier_model=='Random_Forest':
                model = RandomForestClassifier(n_estimators=200, class_weight="balanced")
                model.fit(X_train, y_train)
            elif classifier_model=='Naive_Bayes':
                model = MultinomialNB()
                model.fit(X_train, y_train)
            elif classifier_model=='SVM':
                model = SVC(class_weight="balanced")
                model.fit(X_train, y_train)

        elif balanced==False:
            if classifier_model=='Decision_Tree':
                model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, min_samples_split=2)
                model.fit(X_train, y_train)
            elif classifier_model=='Random_Forest':
                model = RandomForestClassifier(n_estimators=200)
                model.fit(X_train, y_train)
            elif classifier_model=='Naive_Bayes':
                model = MultinomialNB()
                model.fit(X_train, y_train)
            elif classifier_model=='SVM':
                model = SVC()
                model.fit(X_train, y_train)

        return model
