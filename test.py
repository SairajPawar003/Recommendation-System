# from jnius import autoclass

# # Example: Access a Java class
# System = autoclass('java.lang.System')
# System.out.println('Hello from Java!')
# from transformers import is_torch_npu_available

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline is only for Jupyter Notebooks

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# Download NLTK dependencies if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize necessary NLTK objects
lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)

# Install Python-Terrier and initialize
import pyterrier as pt
if not pt.started():
    pt.init()

# Install Neuspell (instructions are detailed below)
# Ensure you've followed the necessary steps to clone the Neuspell repository and install it
# pip install git+https://github.com/neuspell/neuspell.git

from neuspell import BertChecker
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

# LOAD DATASET
# Ensure the dataset is available in the correct directory
# Replace with the correct path to the dataset file
try:
    df = pd.read_csv('flipkart_com-ecommerce_sample.csv')
    print(df.head())
except FileNotFoundError:
    print("Dataset file 'flipkart_com-ecommerce_sample.csv' not found. Please ensure it is in the correct directory.")
