from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# To-do: Create new functions named 'JustificationMinerKMeans', 'JustificationMinerAgglomerative', etc.
# DONE/To-do: Move read txt function out into a separate function from justification mining
# Code to use this in python env is import justification_miners as jm // jm.JustificationMiner(arguments)


# def setInput():
#     n = input("Enter file name of text data: ")
#     return n

def read_file():
    overview_file = input("Enter txt file name of data: ")
    with open(overview_file, 'r') as file:
        initial_corpus = file.read()
    corpus = initial_corpus.split('. ')
    return corpus
#corpustwo = read_file()


# with open('JWN_Nordstrom_MDNA_overview_2017.txt', 'r') as file:
#     initial_corpus = file.read()
# corpus = initial_corpus.split('. ')

def JustificationMiner(corpus, clustering_model=['Kmeans','Agglomerative'], num_clusters=5, save_data=False):
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
    print("Justifications:")
    print(justifications)
    print("")

    if save_data==True:
        all_data_df.to_csv('all_data_df.csv')
        new_df.to_csv('new_df.csv')

    # Print Clusters
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        #print(cluster_id)
    #print(clustered_sentences)
    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i+1)
        print(cluster)
        print("")

    return justifications

#justificationstwo = JustificationMiner(corpustwo, clustering_model='Agglomerative', num_clusters=5, save_data=False)

#####################
#####################
# corpus = corpustwo
# embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
# corpus_embeddings = embedder.encode(corpus)
#
# # Perform KMeans clustering
# num_clusters = 5
# # clustering_model = KMeans(n_clusters=num_clusters)
# clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
# clustering_model.fit(corpus_embeddings)
# # labels = clustering_model.predict(corpus_embeddings)
# cluster_assignment = clustering_model.labels_
# # cluster_centres = clustering_model.cluster_centers_
# clustered_embeddings = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
#     clustered_embeddings[cluster_id].append(corpus_embeddings[sentence_id])
# # for cluster in clustered_embeddings:
# #     print(len(cluster[0]))
# # print(clustered_embeddings)
# cluster_centres = [np.mean(cluster, axis=0) for cluster in clustered_embeddings]
# # print(len(cluster_centres[0]))
# # print(len(cluster_centres[1]))
# # print(len(cluster_centres[2]))
# # print(len(cluster_centres[3]))
# # print(len(cluster_centres[4]))
#
# # Create DataFrame with all data
# cluster_centres_list = [cluster_centres[label] for label in cluster_assignment]
# # cluster_centres_list = []
# # for label in cluster_assignment:
# #     cluster_centres_list.append(cluster_centres[label])
# all_data_df = pd.DataFrame()
# all_data_df['sentences'] = corpus
# all_data_df['embeddings'] = corpus_embeddings
# all_data_df['labels'] = cluster_assignment
# all_data_df['cluster_centres'] = cluster_centres_list
#
# cosine_similarities_list = [cosine_similarity(x.reshape(1,-1),y.reshape(1,-1)) for x, y in zip(all_data_df['embeddings'], all_data_df['cluster_centres'])]
# cosine_similarities_list_final = [cosines[0][0] for cosines in cosine_similarities_list]
# all_data_df['cosine_similarities_to_centre'] = cosine_similarities_list_final
# new_df = all_data_df.sort_values(['labels', 'cosine_similarities_to_centre'], ascending=False).drop_duplicates(['labels'])
# justifications = new_df['sentences'].tolist()
# print("Justifications:")
# print(justifications)
# print("")
#
# if save_data==True:
#     all_data_df.to_csv('all_data_df.csv')
#     new_df.to_csv('new_df.csv')
#
# # Print Clusters
# clustered_sentences = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
#     clustered_sentences[cluster_id].append(corpus[sentence_id])
#     #print(cluster_id)
# #print(clustered_sentences)
# for i, cluster in enumerate(clustered_sentences):
#     print("Cluster ", i+1)
#     print(cluster)
#     print("")
