from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import csv


with open('JWN_Nordstrom_MDNA_overview_2017.csv', 'r') as f:
    reader = csv.reader(f)
    initial_corpus = list(reader)
print(initial_corpus)
corpus = initial_corpus.split('. ')

embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

# Corpus with example sentences
# corpus = ['A man is eating a food.',
#           'A man is eating a piece of bread.',
#           'A man is eating pasta.',
#           'The girl is carrying a baby.',
#           'The baby is carried by the woman',
#           'A man is riding a horse.',
#           'A man is riding a white horse on an enclosed ground.',
#           'A monkey is playing drums.',
#           'Someone in a gorilla costume is playing a set of drums.',
#           'A cheetah is running behind its prey.',
#           'A cheetah chases prey on across a field.']

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
clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")
