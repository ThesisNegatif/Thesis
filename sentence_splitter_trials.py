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
nltk.download('wordnet')
from nltk import tokenize

# alphabets= "([A-Za-z])"
# prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
# suffixes = "(Inc|Ltd|Jr|Sr|Co)"
# starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
# acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
# websites = "[.](com|net|org|io|gov)"
#
# def split_into_sentences(text):
#     text = " " + text + "  "
#     text = text.replace("\n"," ")
#     text = re.sub(prefixes,"\\1<prd>",text)
#     text = re.sub(websites,"<prd>\\1",text)
#     if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
#     text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
#     text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
#     text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
#     text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
#     text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
#     text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
#     text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
#     if "”" in text: text = text.replace(".”","”.")
#     if "\"" in text: text = text.replace(".\"","\".")
#     if "!" in text: text = text.replace("!\"","\"!")
#     if "?" in text: text = text.replace("?\"","\"?")
#     text = text.replace(".",".<stop>")
#     text = text.replace("?","?<stop>")
#     text = text.replace("!","!<stop>")
#     text = text.replace("<prd>",".")
#     sentences = text.split("<stop>")
#     sentences = sentences[:-1]
#     sentences = [s.strip() for s in sentences]
#     return sentences

def read_file_nltk():
    """
    Args:
        None
    Returns:
        corpus(str): Full text corpus read in as a string.
    """
    #overview_file = input("Enter txt file name of data: ")
    with open('JWN_Nordstrom_MDNA_overview_2017.txt', 'r') as file:
        initial_corpus = file.read()
    corpus = tokenize.sent_tokenize(initial_corpus)
    return corpus
# file to be read in line 29 is: 'JWN_Nordstrom_MDNA_overview_2017.txt'
corpustwo = read_file_nltk()
# NLTK's sent_tokenizer works much better than the custom split_into_sentences function above

# def read_file_custom():
#     """
#     Args:
#         None
#     Returns:
#         corpus(str): Full text corpus read in as a string.
#     """
#     #overview_file = input("Enter txt file name of data: ")
#     with open('JWN_Nordstrom_MDNA_overview_2017.txt', 'r') as file:
#         initial_corpus = file.read()
#     corpus = split_into_sentences(initial_corpus)
#     return corpus
# # file to be read in line 29 is: 'JWN_Nordstrom_MDNA_overview_2017.txt'
# corpusthree = read_file_custom()

# from nltk import tokenize
# p = "Good morning Dr. Adams. The patient is waiting for you in room number 3."
#
# tokenize.sent_tokenize(p)
# ['Good morning Dr. Adams.', 'The patient is waiting for you in room number 3.']

# print(corpustwo)
# print(corpusthree)
