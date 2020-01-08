from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


with open('JWN_Nordstrom_MDNA_overview_2017.txt', 'r') as file:
    initial_corpus = file.read()
#print(initial_corpus)


corpus = initial_corpus.split('. ')

embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

corpus_embeddings = embedder.encode(corpus)

#for sentence, embedding in zip(corpus, corpus_embeddings):
#    print("Sentence:", sentence)
#    print("Embedding:", embedding)
#    print("")

# Perform kmean clustering
#num_clusters = 5
#whitened_corpus = scipy.cluster.vq.whiten(corpus_embeddings)
#code_book, _ = scipy.cluster.vq.kmeans(whitened_corpus, num_clusters)
#cluster_assignment, _ = scipy.cluster.vq.vq(whitened_corpus, code_book)

# Perform kmean clustering
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
labels = clustering_model.fit_predict(corpus_embeddings)
cluster_assignment = clustering_model.labels_
cluster_centre = clustering_model.cluster_centers_

print(cluster_centre.shape)
print(labels.shape)
print(len(corpus_embeddings))
print(corpus_embeddings[0].shape)

df1 = pd.DataFrame()
df1['sentences'] = corpus_embeddings
df1['labels'] = labels

#print(df1.head())
print(df1.shape)
##print(df1['sentences'][df1['labels']==1])

print(cluster_centre)
print(df1.head())


#print(df1['sentences'][df1['labels']==0][0])

repeatnumber = len(df1['sentences'][df1['labels']==0])
print(repeatnumber)

similarities = cosine_similarity(df1['sentences'][df1['labels']==0], cluster_centre[0])
#similarities = cosine_similarity(df1['sentences'][df1['labels']==0][0], cluster_centre[0])
print(similarities)

#corpus_embeddings is a list of arrays, labels is an array with the labels for each sentence
#cluster_centre is a list of 5 arrays for the 5 centres of each cluster
#cluster_centre is shape 5,768. labels has 29 points, len of corpus_embeddings is 29 and
#shape of each corpus_embedding array is 768,

# clustered_sentences = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
#     clustered_sentences[cluster_id].append(corpus[sentence_id])

# for i, cluster in enumerate(clustered_sentences):
#     print("Cluster ", i+1)
#     print(cluster)
#     print("")
