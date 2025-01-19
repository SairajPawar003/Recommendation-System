import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline  its only for jupitor 
import string 
import re 
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)

#install python-terrier
import pyterrier as pt 
if not pt.started():
    pt.init()

#pip install -U sentence-transformers
#pip install neuspell
'''following processes is complicated but you like it
you need to download git from https://git-scm.com/
then install it check version by using git --version command then run 
git clone https://github.com/neuspell/neuspell 
cd neuspell
pip install -e .
'''
#pip install -e neuspell/
#git clone https://github.com/neuspell/neuspell; cd neuspell 

# import os 
# os.chdir("/content/neuspell")
'''following code is tricky find extras-requirements.txt file in your c drive '''
#pip install -r/content/neuspell/extras-requirements.txt
#python -m spacy download en_core_web_sm
#Unzipping the multi-linguistic packages 
'''doenload wget.exe'''
# !wget https://storage.googleapis.com/bert_model/2018_11_23/multi_cased_L-12_H-768_A-12.zip
# !unzip *.zip

#importing neuspell
from neuspell import BertChecker
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformer/bert-base-nli-mean-tokens')

#LOAD DATASET 
df = pd.read_csv('flipkart_com-ecommerce_sample.csv')
print(df.head())