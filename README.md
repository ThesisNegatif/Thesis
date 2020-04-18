Repository contains code written for the research investigation of my doctoral thesis at the University of Oxford.

The main files are justification_miners.py, classifier_trainers.py and sentiment_analysers.py.

In order to mine justifications using SBERT:

```
from justification_miners import JustificationMiner

J = JustificationMiner()

J.mine_justifications(example_document, clustering_model='DBSCAN', num_clusters=5, save_data=False, embedder='NLI')
# where example_document is the document on which you want to perform justification mining
```

Testing data used in my thesis is contained in final_all_mdna_data.csv, which has manually parsed information from the MD&A subsections of numerous firms listed on the S&P500.
