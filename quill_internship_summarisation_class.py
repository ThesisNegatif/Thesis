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
import pickle
# nltk.download('wordnet') # Needs to be uncommented and installed once if wordnet isn't already installed
from nltk import tokenize

class SessionSummariser(object):
    def mine_sentences(self, string_text, clustering_model=['Kmeans','Agglomerative'], num_clusters=5, save_data=False):
        """
        Args:
            string_text(str): Text corpus on which to perform mining.
            clustering_model(str): Clustering model to use. Options are 'Kmeans' and 'Agglomerative'.
            num_clusters(int): Number of clusters returned by clustering models. Defaults to 5.
            save_data(bool): Whether to save Pandas dataframes of both the full data and max cosines
                similarity scores data. Saves in two new csv files within same directory. Defaults to False.
        Returns:
            justifications(list): Full list of justification sentences returned by the miner.
        """
        # Split Sentences
        corpus = tokenize.sent_tokenize(string_text)

        # Extract sentence embeddings
        embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
        corpus_embeddings = embedder.encode(corpus)

        # Perform KMeans or Agglomerative clustering
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

        # Print Clusters
        # clustered_sentences = [[] for i in range(num_clusters)]
        # for sentence_id, cluster_id in enumerate(cluster_assignment):
        #     clustered_sentences[cluster_id].append(corpus[sentence_id])
        # for i, cluster in enumerate(clustered_sentences):
        #     print("Cluster ", i+1)
        #     print(cluster)
        #     print("")

        return justifications

trial_text = 'Nordstrom is a leading fashion specialty retailer offering apparel, shoes, cosmetics and accessories for women, men, young adults and children. We offer an extensive selection of high-quality brand-name and private label merchandise through our various channels, including Nordstrom U.S. and Canada full-line stores, Nordstrom.com, Nordstrom Rack stores, Nordstromrack.com/HauteLook, Trunk Club clubhouses and TrunkClub.com, Jeffrey boutiques and Last Chance clearance stores. As of January 28, 2017, our stores are located in 40 states throughout the United States and in three provinces in Canada. Our customers can participate in our Nordstrom Rewards loyalty program which allows them to earn points based on their level of spending. We also offer our customers a variety of payment products and services, including credit and debit cards. Our 2016 earnings per diluted share of $2.02, which included a goodwill impairment charge of $1.12, exceeded our outlook of $1.70 to $1.80. Our results were driven by continued operational efficiencies in inventory and expense execution and demonstrated our team’s speed and agility in responding to changes in business conditions. We reached record sales of $14.5 billion for the year, reflecting a net sales increase of 2.9% and comparable sales decrease of 0.4% primarily driven by full-line stores. We achieved the following milestones in multiple growth areas. Our expansion into Canada where we currently have five full-line stores, including two that opened last fall, contributed total sales of $300 in 2016. Nordstrom.com sales reached over $2.5 billion, representing approximately 25% of full-price sales. Our off-price business reached $4.5 billion, with growth mainly driven by our online net sales increase of 32% and 21 new store openings. Off-price continues to be our largest source of new customers, gaining approximately 6 million in 2016. Our expanded Nordstrom Rewards program, which launched in the second quarter, drove a strong customer response with 3.7 million customers joining through our non-tender offer. We ended the year with a total of 7.8 million active Nordstrom Rewards customers. Our working capital improvements contributed to the $1.6 billion in operating cash flow and $0.6 billion in free cash flow. From a merchandising perspective, we strive to offer a curated selection of the best brands. As we look for new opportunities through our vendor partnerships, we will continue to be strategic and pursue partnerships that are similar to our portfolio and maintain relevance with our customers by delivering newness. Our strategies around product differentiation include our ongoing efforts to grow limited distribution brands such as Ivy Park, J.Crew and Good American, in addition to our Nordstrom exclusive offering. In 2016, we made focused efforts to improve our productivity, particularly around our technology, supply chain and marketing. In technology, we increased the productivity of delivering features to enhance the customer experience. In supply chain, we focused on overall profitability by reducing split shipments and editing out less profitable items online. In marketing, we strengthened our capabilities around digital engagement so that we reach customers in a more efficient and cost-effective manner. Through these efforts, we made significant progress in improving operational efficiencies, reflected by moderated expense growth of 10% in these three key areas, relative to an annual average of 20% over the past five years. With customer expectations changing faster than ever, it is important that we remain focused on the customer. Moving forward, we believe our strategies give us a platform for enhanced capabilities to better serve customers and increase market share. Our obsession with our customers keeps us focused on speed, convenience and personalization. We have good momentum in place and will continue to make changes to ensure we are best serving customers and improving our business now and into the future.'

m = SessionSummariser()
summarised_sentences = m.mine_sentences(trial_text, clustering_model='Kmeans', num_clusters=5, save_data=False)
print(summarised_sentences)
