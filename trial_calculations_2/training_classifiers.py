import pandas as pd
import json
import nltk
import os
import random
import re
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

len(list_data)

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

# print(token_ids[:10])
# print(len(filtered_words))
# print(len(filtered))
# print(len(token_ids))
# print(sentiments[:10])


X_train, X_test, y_train, y_test = train_test_split(token_ids, sentiments, test_size=0.2, random_state=42)
for i, sentence in enumerate(X_train):
    if len(sentence) <=40:
        X_train[i] = ((40-len(sentence)) * [0] + sentence)
    elif len(sentence) > 40:
        X_train[i] = sentence[:40]

for i, sentence in enumerate(X_test):
    if len(sentence) <=40:
        X_test[i] = ((40-len(sentence)) * [0] + sentence)
    elif len(sentence) > 40:
        X_test[i] = sentence[:40]

# for sentence in X_test:
#     print(len(sentence))

# Decision Tree
# model_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, min_samples_split=2)
# model_dt.fit(X_train, y_train)
#
# y_train_pred_dt = model_dt.predict(X_train)
# y_test_pred_dt = model_dt.predict(X_test)
#
# train_accuracy = accuracy_score(y_train, y_train_pred_dt)
# test_accuracy = accuracy_score(y_test, y_test_pred_dt)
# print('The training accuracy is', train_accuracy)
# print('The test accuracy is', test_accuracy)
# print('Accuracy score: ', format(accuracy_score(y_test, y_test_pred_dt)))
# print('Precision score: ', format(precision_score(y_test, y_test_pred_dt, average='weighted')))
# print('Recall score: ', format(recall_score(y_test, y_test_pred_dt, average='weighted')))
# print('F1 score: ', format(f1_score(y_test, y_test_pred_dt, average='weighted')))

# Random Forest
random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, y_train)

train_predictions_rf = random_forest.predict(X_train)
predictions_rf = random_forest.predict(X_test)
train_accuracy_rf = accuracy_score(y_train, train_predictions_rf)

print('The training accuracy is:', format(accuracy_score(y_train, train_predictions_rf)))
print('Test Accuracy score: ', format(accuracy_score(y_test, predictions_rf)))
print('Precision score: ', format(precision_score(y_test, predictions_rf, average='weighted')))
print('Recall score: ', format(recall_score(y_test, predictions_rf, average='weighted')))
print('F1 score: ', format(f1_score(y_test, predictions_rf, average='weighted')))

# Naive Bayes
# naive_bayes = MultinomialNB()
# naive_bayes.fit(X_train, y_train)
#
# nb_predictions_train = naive_bayes.predict(X_train)
# nb_predictions_test = naive_bayes.predict(X_test)
#
# print('The training accuracy is:', format(accuracy_score(y_train, nb_predictions_train)))
# print('Test Accuracy score: ', format(accuracy_score(y_test, nb_predictions_test)))
# print('Precision score: ', format(precision_score(y_test, nb_predictions_test, average='weighted')))
# print('Recall score: ', format(recall_score(y_test, nb_predictions_test, average='weighted')))
# print('F1 score: ', format(f1_score(y_test, nb_predictions_test, average='weighted')))

# SVM
# SVM = SVC()
# SVM.fit(X_train, y_train)
#
# svm_predictions_train = SVM.predict(X_train)
# svm_predictions_test = SVM.predict(X_test)
#
# print('The training accuracy is:', format(accuracy_score(y_train, svm_predictions_train)))
# print('Test Accuracy score: ', format(accuracy_score(y_test, svm_predictions_test)))
# print('Precision score: ', format(precision_score(y_test, svm_predictions_test, average='weighted')))
# print('Recall score: ', format(recall_score(y_test, svm_predictions_test, average='weighted')))
# print('F1 score: ', format(f1_score(y_test, svm_predictions_test, average='weighted')))
