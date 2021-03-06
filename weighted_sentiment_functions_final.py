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

# Code to use this in python env is import justification_miners as jm // jm.JustificationMiner(arguments)
# To-do: Want a way to retrieve cluster_centres, clusters, dataframe.head, justifications

# MAYBE but everything into one giant function so that
# when you enter in a document, it will mine and score the sentiment of the justifications
# Try this in a new file
# Need 3 functions (weighted, unweighted and whole document) then can put these directly
# Into a dataframe on jupyter
# ACTUALLY might be easier to create a new function that just calls on
# the other functions one by one. This is probably how main files work but for now
# can just do whatever works. this way we can keep training the model separate
# and have the model trained separately and then have the main function call on
# a trained model as a parameter. The train_model function could include
# opening the file, processing and tokenising it and then fitting it to a model
# with the final model being returned.

# Function to be made into a class is below.
def JustificationMiner(string_text, clustering_model=['Kmeans','Agglomerative','DBSCAN','MeanShift'], num_clusters=5, save_data=False, embedder=['Wiki','NLI']):
    """
    Args:
        string_text(str): Text on which to perform Justification Mining.
        clustering_model(str): Clustering model to use. Options are 'Kmeans' and 'Agglomerative'.
        num_clusters(int): Number of clusters returned by clustering models. Defaults to 5.
        save_data(bool): Whether to save Pandas dataframes of both the full data and max cosines
            similarity scores data. Saves in two new csv files within same directory. Defaults to False.
    Returns:
        justifications(list): Full list of justification sentences returned by the miner.
    """
    # To-do: Include processing steps here if required (not required)
    # Email from Dr. Reimers says no processing beforehand is required.
    # Split Sentences
    corpus = tokenize.sent_tokenize(string_text)
    # Extract sentence embeddings
    # Also try 'bert-base-nli-mean-tokens' instead of Wikipedia for comparison
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
    elif clustering_model=='MeanShift':
        clustering_model = MeanShift()
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        cluster_centres = clustering_model.cluster_centers_
    elif clustering_model=='DBSCAN':
        clustering_model = DBSCAN(eps=0.5)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        # Note: because -1 is for noisy samples, it is difficult to use DBSCAN without a random noisy cluster returned
        # new_array = cluster_assignment[cluster_assignment != -1]
        # number_of_clusters = len(set(new_array))
        # cluster_assignment = cluster_assignment[cluster_assignment != -1]
        number_of_clusters = len(set(cluster_assignment))
        clustered_embeddings = [[] for i in range(number_of_clusters)]
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
    # clustered_sentences = [[] for i in range(num_clusters)]
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     clustered_sentences[cluster_id].append(corpus[sentence_id])
        #print(cluster_id)
    #print(clustered_sentences)
    # for i, cluster in enumerate(clustered_sentences):
    #     print("Cluster ", i+1)
    #     print(cluster)
    #     print("")

    return justifications


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

# Make below into a new function called def numeric_tokenize()
def create_X_train():
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
    sentiments = [sentence['sentiment'] for sentence in dictionary_data['data']] # This is returned as sentiments when not using balance function below

    tokenized = [preprocess(message) for message in messages] # Previously called just tokenized or tokenized_unbalanced

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

    bow = Counter([j for i in tokenized for j in i]) #[word for sentence in tokenized for word in sentence] Should the for loops be in reverse order? No, it raises NameError
    # Try both ways above on some dummy data in jupyter
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
    return vocab, X_train, sentiments # or sentiments_balanced when using balancing function

def create_X_test(sentences, vocab):
    tokenized = [preprocess(sentence) for sentence in sentences]
    filtered = [[word for word in sentence if word in vocab.keys()] for sentence in tokenized] # X_test filtered to only words in training vocab
    # Alternate method with functional programming:
    # filtered = [list(filter(lambda a: a in vocab.keys(), sentence)) for sentence in tokenized]
    token_ids = [[vocab[word] for word in sentence] for sentence in filtered] # Numericise data
    # print('token ids:', token_ids)
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

def train_classifier(classifier_model=['Decision_Tree','Random_Forest', 'Naive_Bayes', 'SVM']):
    vocab, X_train, sentiments = create_X_train()
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

###### THIS IS IMPORTANT FOR SAVING THE TRAINED MODEL #####
# # Comment/Uncomment following 4 lines to retrain the classifier
# trained_model = train_classifier(classifier_model='Random_Forst')
# # save the model to disk
# filename = 'finalised_classifier_model.sav'
# pickle.dump(trained_model, open(filename, 'wb'))

# # load the model from disk
# filename = 'finalised_classifier_model.sav'
# model = pickle.load(open(filename, 'rb'))
############################################################

def create_X_train_embeddings(embedder=['Wiki','NLI']):
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
    # Also try 'bert-base-nli-mean-tokens' instead of Wikipedia for comparison
    if embedder=='Wiki':
        embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
    elif embedder=='NLI':
        embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    corpus_embeddings = embedder.encode(messages)
    X_train_embeddings = corpus_embeddings
    return X_train_embeddings, sentiments_embeddings

def create_X_test_embeddings(sentences, embedder=['Wiki','NLI']):
    # Extract sentence embeddings
    # Also try 'bert-base-nli-mean-tokens' instead of Wikipedia for comparison
    if embedder=='Wiki':
        embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
    elif embedder=='NLI':
        embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    corpus_embeddings = embedder.encode(sentences)
    X_test_embeddings = corpus_embeddings
    return X_test_embeddings

def train_classifier_embeddings(classifier_model=['Decision_Tree','Random_Forest', 'Naive_Bayes', 'SVM'], balanced=True, embedder=['Wiki','NLI']):
    X_train, sentiments = create_X_train_embeddings(embedder=embedder)
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



# Below are the main functions for getting sentiment scores
def get_mined_sentiment(corpus, vocab, model, clustering_model=['Kmeans','Agglomerative','DBSCAN','MeanShift'], numericiser=['BoW','SBERT'], embedder=['Wiki', 'NLI'], aggregation=['Mean','Mode','MeanRounded']):
    justifications = JustificationMiner(corpus, clustering_model=clustering_model, num_clusters=5, save_data=False, embedder=embedder)
    if numericiser=='BoW':
        X_test = create_X_test(justifications, vocab)
        justifications_predictions = model.predict(X_test)
    elif numericiser=='SBERT':
        X_test = create_X_test_embeddings(justifications, embedder=embedder)
        justifications_predictions = model.predict(X_test)
    # print(justificationstwo)
    # print(justifications_predictions)
    # Weighted calculation below: Note, this would be the unweighted sentiment to correlate to stock returns
    # Note predictions are divided by 2 so values are between 0 and 1 for logit function later
    # justification_weighted_score = (justifications_predictions/2.0).mean() # for 0-1
    if aggregation=='Mean':
        justification_score = (justifications_predictions).mean()
    elif aggregation=='Mode':
        justification_score = stats.mode(justifications_predictions)[0][0]
    elif aggregation=='MeanRounded':
        justification_score = (justifications_predictions).mean().round()
    # print(justification_weighted_scores)
    return justification_score




def get_weighted_sentiment(corpus, vocab, model, clustering_model=['Kmeans','Agglomerative','DBSCAN','MeanShift'], numericiser=['BoW','SBERT'], embedder=['Wiki', 'NLI']):
    justifications = JustificationMiner(corpus, clustering_model=clustering_model, num_clusters=5, save_data=False, embedder=embedder)
    if numericiser=='BoW':
        X_test = create_X_test(justifications, vocab)
        justifications_predictions = model.predict(X_test)
    elif numericiser=='SBERT':
        X_test = create_X_test_embeddings(justifications, embedder=embedder)
        justifications_predictions = model.predict(X_test)
    # print(justificationstwo)
    # print(justifications_predictions)
    # Weighted calculation below: Note, this would be the unweighted sentiment to correlate to stock returns
    # Note predictions are divided by 2 so values are between 0 and 1 for logit function later
    # justification_weighted_score = (justifications_predictions/2.0).mean() # for 0-1
    justification_weighted_score = (justifications_predictions).mean()
    # print(justification_weighted_scores)
    return justification_weighted_score

# Might need to modify this, instead round each value maybe?
# In regular sent analysis you get 0, 0.5, 1. With averaging justifications you get between 0 and 1, i.e. scaled/weighted.
# Below is more like weighted mode aggregated. Whereas above 'weighted' function is like weighted mean aggregated.
# But mode doesn't make much sense because it gets rid of the weight, so it is more like checking if 5 mined justifications can
# capture the full sentiment of the entire document.
# Read what you wrote in your thesis and then clarify all these terms
def get_unweighted_sentiment(corpus, vocab, model, clustering_model=['Kmeans','Agglomerative','DBSCAN','MeanShift'], numericiser=['BoW','SBERT'], embedder=['Wiki', 'NLI']):
    justifications = JustificationMiner(corpus, clustering_model=clustering_model, num_clusters=5, save_data=False, embedder=embedder)
    if numericiser=='BoW':
        X_test = create_X_test(justifications, vocab)
        justifications_predictions = model.predict(X_test)
    elif numericiser=='SBERT':
        X_test = create_X_test_embeddings(justifications, embedder=embedder)
        justifications_predictions = model.predict(X_test)
    # print(justificationstwo)
    # print(justifications_predictions)
    # Weighted calculation below: Note, this would be the unweighted sentiment to correlate to stock returns
    # Note predictions are divided by 2 so values are between 0 and 1 for logit function later
    # justification_unweighted_score = stats.mode(justifications_predictions/2.0)[0][0] # for 0-1
    justification_unweighted_score = stats.mode(justifications_predictions)[0][0]
    # print(justification_unweighted_scores)
    return justification_unweighted_score

def get_full_sentiment(corpus, vocab, model, numericiser=['BoW','SBERT'], embedder=['Wiki','NLI'], aggregation=['Mean','Mode','MeanRounded']):
    sentences = tokenize.sent_tokenize(corpus)
    if numericiser=='BoW':
        X_test = create_X_test(sentences, vocab)
        predictions = model.predict(X_test)
    elif numericiser=='SBERT':
        X_test = create_X_test_embeddings(sentences, embedder=embedder)
        predictions = model.predict(X_test)
    # print(predictions)
    # full_score_mean_unrounded = (predictions/2.0).mean()
    if aggregation=='Mean':
        full_score = (predictions).mean()
    elif aggregation=='Mode':
        full_score = stats.mode(predictions)[0][0]
    elif aggregation=='MeanRounded':
        full_score = (predictions).mean().round()
    return full_score

def get_full_sentiment_mean_unrounded(corpus, vocab, model, numericiser=['BoW','SBERT'], embedder=['Wiki','NLI']):
    sentences = tokenize.sent_tokenize(corpus)
    if numericiser=='BoW':
        X_test = create_X_test(sentences, vocab)
        predictions = model.predict(X_test)
    elif numericiser=='SBERT':
        X_test = create_X_test_embeddings(sentences, embedder=embedder)
        predictions = model.predict(X_test)
    # print(predictions)
    # full_score_mean_unrounded = (predictions/2.0).mean()
    full_score_mean_unrounded = (predictions).mean()
    return full_score_mean_unrounded

def get_full_sentiment_mean(corpus, vocab, model, numericiser=['BoW','SBERT'], embedder=['Wiki','NLI']):
    sentences = tokenize.sent_tokenize(corpus)
    if numericiser=='BoW':
        X_test = create_X_test(sentences, vocab)
        predictions = model.predict(X_test)
    elif numericiser=='SBERT':
        X_test = create_X_test_embeddings(sentences, embedder=embedder)
        predictions = model.predict(X_test)
    # print(predictions)
    # full_score_mean = (predictions/2.0).mean().round()
    full_score_mean = (predictions).mean().round()
    return full_score_mean

def get_full_sentiment_mode(corpus, vocab, model, numericiser=['BoW','SBERT'], embedder=['Wiki','NLI']):
    sentences = tokenize.sent_tokenize(corpus)
    if numericiser=='BoW':
        X_test = create_X_test(sentences, vocab)
        predictions = model.predict(X_test)
    elif numericiser=='SBERT':
        X_test = create_X_test_embeddings(sentences, embedder=embedder)
        predictions = model.predict(X_test)
    # print(predictions)
    # full_score_mode = stats.mode(predictions/2.0)[0][0]
    full_score_mode = stats.mode(predictions)[0][0]
    return full_score_mode
