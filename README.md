# Repository contains code written for the research investigation of my doctoral thesis at the University of Oxford.

The main files are justification_miners.py, classifier_trainers.py and sentiment_analysers.py.

In order to mine justifications using SBERT (reference: https://github.com/UKPLab/sentence-transformers) embeddings and clustering algorithms:

```
from justification_miners import JustificationMiner

J = JustificationMiner()

J.mine_justifications(example_document, clustering_model='DBSCAN', num_clusters=5, save_data=False, embedder='NLI')

# where example_document is the document on which you want to perform justification mining
```

Check the docstrings for lists of clustering algorithms and SBERT datasets.

To create classifier training and testing inputs using SBERT embeddings to numericise text:

```
from classifier_trainers import TrainClassifier

C = TrainClassifier()

C.create_X_test_embeddings(test_sentences, embedder='Wiki') # where test_sentences are the sentences to be numericised
```

Note, the file by default utilises the training data from Malo et al. (2013) referenced in my thesis, but this can be changed by specifying a different training dataset (in .csv format).

Testing data used in my thesis is contained in final_all_mdna_data.csv, which has manually parsed information from the MD&A subsections of numerous firms listed on the S&P500.
