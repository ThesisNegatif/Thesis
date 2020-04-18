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

# Extract sentence embeddings
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

# Perform KMeans clustering
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
#labels = clustering_model.predict(corpus_embeddings)
cluster_assignment = clustering_model.labels_
cluster_centres = clustering_model.cluster_centers_

# print(cluster_centres.shape)
# print(labels.shape)
# print(len(corpus_embeddings))
# print(corpus_embeddings[0].shape)

#cluster_centres_list = [cluster_centres[label] for label in cluster_assignment]
cluster_centres_list = []
for label in cluster_assignment:
    cluster_centres_list.append(cluster_centres[label])
#print(cluster_centres_list[1].shape)

all_data_df = pd.DataFrame()
all_data_df['sentences'] = corpus
all_data_df['embeddings'] = corpus_embeddings
all_data_df['labels'] = cluster_assignment
all_data_df['cluster_centres'] = cluster_centres_list
# print(all_data_df['embeddings'][1].shape)
# print(all_data_df['cluster_centres'][1].shape)

#DOESN'T WORK: all_data_df['cosine_similarities_to_centre'] = cosine_similarity(all_data_df['embeddings'].reshape(1,-1), all_data_df['cluster_centres'].reshape(1,-1))
##LINE BELOW IS THE ORIGINAL METHOD TO ADD COSINES TO DATAFRAME THAT WORKS
##all_data_df['cosine_similarities_to_centre'] = [cosine_similarity(x.reshape(1,-1),y.reshape(1,-1)) for x, y in zip(all_data_df['embeddings'], all_data_df['cluster_centres'])]
cosine_similarities_list = [cosine_similarity(x.reshape(1,-1),y.reshape(1,-1)) for x, y in zip(all_data_df['embeddings'], all_data_df['cluster_centres'])]
cosine_similarities_list_final = [cosines[0][0] for cosines in cosine_similarities_list]
all_data_df['cosine_similarities_to_centre'] = cosine_similarities_list_final
#print(len(cosine_similarities_list))

# for cosines in cosine_similarities_list:
#     print(cosines[0][0])
# print(cosine_similarities_list[0][0][0])

#print(all_data_df)
#all_data_df.to_csv('all_data_df.csv')

# Tried below 3 methods on dataframe with toy data and same columns and all worked, but on
# actual data only the last method worked.
# Method 1: using idxmax() function
# print(all_data_df.loc[all_data_df.groupby(["labels"])["cosine_similarities_to_centre"].idxmax()])

# Method 2: groupby, max and getting index
# idx = all_data_df.groupby(['labels'])['cosine_similarities_to_centre'].transform(max) == all_data_df['cosine_similarities_to_centre']
# print(all_data_df[idx])

# METHOD BELOW WORKS!!!!! method using sort_values
new_df = all_data_df.sort_values(['labels', 'cosine_similarities_to_centre'], ascending=False).drop_duplicates(['labels'])
print(len(new_df['sentences'].tolist()))
#new_df.to_csv('new_df.csv')

##### BELOW THIS LINE IS UNCESSARY STUFF #####
##############################################

#for i in range(num_clusters):
# df_labels = all_data_df[all_data_df['labels'] == 1]
# print(df_labels)
# print(df_labels.index)
# print(df_labels[df_labels['cosine_similarities_to_centre'].max()])



# max_index = df_labels.loc[df_labels.idxmax(),'cosine_similarities_to_centre']
# print(max_index)

#new_df = all_data_df.groupby(['labels']).max()
#print(new_df)
#new_df.to_csv('new_df.csv')


# for index, row in all_data_df.iterrows():
#     iforval = cluster_centres[row['labels']]
#     #print(iforval)
#     #All_data_df.at[index, 'cluster_centres'] = iforval
# print(all_data_df.shape)

#print(df1.head())
# print(df1.shape)
##print(df1['sentences'][df1['labels']==1])

# print(cluster_centres)
# print(df1.head())

# print('cluster_assignment', cluster_assignment)


#print(df1['sentences'][df1['labels']==0][0])
#
# repeatnumber = len(df1['sentences'][df1['labels']==0])
# print(repeatnumber)

#similarities = cosine_similarity(df1['sentences'][df1['labels']==0], cluster_centres[0])
#similarities = cosine_similarity(df1['sentences'][df1['labels']==0][0], cluster_centres[0])
#print(similarities)

#after getting similarities use the index to get the original sentence back from original corpus

#corpus_embeddings is a list of arrays, labels is an array with the labels for each sentence
#cluster_centres is a list of 5 arrays for the 5 centres of each cluster
#cluster_centres is shape 5,768. labels has 29 points, len of corpus_embeddings is 29 and
#shape of each corpus_embedding array is 768,

#Code below to print the clusters
# clustered_sentences = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
#     clustered_sentences[cluster_id].append(corpus[sentence_id])
#     #print(cluster_id)
# print(clustered_sentences)
# for i, cluster in enumerate(clustered_sentences):
#     print("Cluster ", i+1)
#     print(cluster)
#     print("")

# Regroup sentence embeddings into 5 clusters
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus_embeddings[sentence_id])

#Calculate cosine similarities
cosine_similarities = [[] for i in range(num_clusters)]
for cluster_id, cluster in enumerate(clustered_sentences):
    for sentence in cluster:
        cosine_similarity_score = cosine_similarity(sentence.reshape(1,-1), cluster_centres[cluster_id].reshape(1,-1))
        cosine_similarities[cluster_id].append(cosine_similarity_score)

# Another method I thought of trying was to put data into key, values of a dictionary
# and then iterate over that, because iterating over dictionaries is generally faster
# However, it would need to be multiple values to each key and the organisation of it
# was too messy.
# Also thought to regroup the sentences into their clusters before calculating cosine-scores
# however, even then, there had to be a way to keep track of the original sentence and not
# just the embedding or cosine score, in order to print the original sentence at the end.

# for cluster in cosine_similarities:
#     print(max(cluster))

#print(cosine_similarities)
#Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
