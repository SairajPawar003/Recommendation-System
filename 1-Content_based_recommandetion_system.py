#Data Manipulating
import pandas as pd
import numpy as np 

#Data Visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 

#NLP for text preprocessing 
import nltk
import scipy 
import re 
from scipy import spatial 
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
tokenizer = ToktokTokenizer

#other Libraries
import gensim 
from gensim.models import Word2Vec

import itertools 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

#remove warning 
import warnings 
warnings.filterwarnings(action='ignore')


def predict_products(text):
    # Getting the index of the input product
    index = product_index[text]
    
    # Pairwise similarity scores
    score_matrix = linear_kernel(T_vec_matrix[index], T_vec_matrix)
    matching_sc = list(enumerate(score_matrix[0]))
    
    # Sort the products based on the similarity score
    matching_sc = sorted(matching_sc, key=lambda x: x[1], reverse=True)
    
    # Getting the scores of the top 10 most similar products (excluding the input product itself)
    matching_sc = matching_sc[1:11]  # Fix: Correct slicing
    
    # Getting product indices of the top matches
    product_indices = [i[0] for i in matching_sc]
    
    # Show the similar products
    return data['product_name'].iloc[product_indices]


# def predict_products(text):
#     #geting index
#     index = product_index[text]
#     #pairwise similarity scores
#     score_matrix = linear_kernel(T_vec_matrix[index], T_vec_matrix)
#     matching_sc = list(enumerate(score_matrix[0]))
#     #sort the product based on the similarity score
#     matching_sc = sorted(matching_sc, key = lambda x:x[1],reverse=True)
#     #Getting Score of 10 most similar products 
#     matching_sc - matching_sc[1:11]
#     #getting product indices
#     product_indices = [i[0] for i in matching_sc]
#     #show the similar product 
#     return data['product_name'].iloc[product_indices]

data = pd.read_csv('flipkart_com-ecommerce_sample.csv')

# print(data.head())
# print(data.describe())
# print(data.info())
# print(data.dtypes)
# print(data.shape)

data['length'] = data['description'].str.len()
# print(data['description'])

# adding new column for the number of words in description befor text preprocessing 
# data['no_of_words'] = data.description.apply(lambda x : len(x.split()))
data['no_of_words'] = data['description'].fillna("").apply(lambda x: len(str(x).split()))
# the following word count for description 

bins = [0,50,75,np.inf]
# data['bins'] = pd.cut(data.no_of_words, bins =[0,100,200,300,500,800,np.inf] , labels =['0-100','100-200','200-500','500-800','>800'])
data['bins'] = pd.cut(
    data['no_of_words'],
    bins=[0, 100, 200, 300, 500, 800, np.inf],  # 6 bin edges, creating 5 bins
    labels=['0-100', '100-200', '200-300', '300-500', '500-800', '>800']  # 6 labels for 6 bins
)

# word_distribution = data.groupby('bins').size().reset_index().rename(columns={0:'word_count'})
# sns.barplot(x='bins', y='word_counts', data= word_distribution).set_title("word distribution per bin")
word_distribution = data.groupby('bins').size().reset_index(name='word_count')

# Plot using Seaborn
sns.barplot(x='bins', y='word_count', data=word_distribution).set_title("Word Distribution per Bin")

# Show the plot
# plt.show()


#ii) Data processing 

# 1) missing values in each column 
missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0:'missing'})
#2) Creating percentage of missing values 
missing['percent'] = missing['missing']/len(data)
#3)sorting values in decending order to see hiest count on the top 
missing.sort_values('percent', ascending=False)

# print(missing)

#iii) data processing by multiple methods 
'''
converting text to lowercase 
removing/replacing the punctuations
removing/replacing the number 
removing the extra whitespaces 
removing the stop word 
Stemming and lemematization 
'''

#removing the punctuations
data['description'] = data['description'].str.replace(r'[^\w\d\s]',' ')

#removing the whitespace between a term with single space
data['description'] = data['description'].str.replace(r'\s+',' ')

# remove leading and trailing whitespace
data['description'] = data['description'].str.replace(r'^\s+|\s+?$','')

#converting to loer case 
data['description'] = data['description'].str.lower()

# print(data['description'].head())

# Removing the stopwords 
stop = stopwords.words('english')
pattern = r'\b(?:{})\b'.format('|'.join(stop))
data['description'] = data['description'].str.replace(pattern,'')
# print(data['description'].head())

# Removing the single characters 
# data['description'] = data['description'].str.replace(r'\s+',' ')
# data['description'] = data['description'].apply(lambda x: " ".join(x for x in x.split() if len(x)>1))
# print(data['description'].head())
# Replace multiple spaces with a single space (explicitly set regex=True)
data['description'] = data['description'].astype(str).str.replace(r'\s+', ' ', regex=True)
# Remove words with length <= 1
data['description'] = data['description'].apply(lambda x: " ".join(word for word in x.split() if len(word) > 1))
# Print the first few rows
# print(data['description'].head())

#removing the domain related to stop word from description 
specific_stop_words = ["rs","flipkart","buy","com","free","day","cash","replacement","guarantee","genuine","key","feature","delivery","products","product","shipping","online","india","stop"]
data['description']=data['description'].apply(lambda x: " ".join(x for x in x.split() if x not in specific_stop_words))
# print(data['description'].head())

# top frequent words after removing domain related stop words 
a = data['description'].str.cat(sep=' ')
word = nltk.tokenize.word_tokenize(a)
word_dist = nltk.FreqDist(word)
# word_dist.plot(10, cumulative=False, title="Top 10 Most Common Words")
word_dist.plot(10,cumulative=False)
print(word_dist.most_common(10))
# plt.show()

#iii) MODEL BULDING 

#1) CONTENT BASED RECOMMENDATION SYSTEM 
#text cleaning process 
data['description'] = data['description'].fillna('')

#define vectorizer
T_vec = TfidfVectorizer(stop_words='english')

#get the vectors 
T_vec_matrix = T_vec.fit_transform(data['description'])

#shape
print(T_vec_matrix.shape)

#reversing the map of indices and products 
product_index = pd.Series(data.index,index=data['product_name']).drop_duplicates()
# print(product_index)

'''In Follwing Steps, everything is wrapped under a single function to make testing easier
1 - obtain index of given product 
2 - obtain cousine similarity score 
3 - sort the scores
4- get the top N result from the list 
5 - out put the product name'''

# # Get recommended products based on user input
# input_product = input("Enter a product name: ")
# try:
#     recommended_products = predict_products(input_product)
#     if not recommended_products.empty:
#         print("Similar Products:\n")
#         for product_name in recommended_products:
#             print(product_name)
#     else:
#         print("No similar products found.")
# except KeyError:
#     print("The entered product name does not exist in the dataset. Please try again.")

# Get recommended products based on user input
input_product = input("Enter a product name: ")##Engage Urge and Urge Combo Set ## Lee Parke Running Shoes

try:
    recommended_products = predict_products(input_product)
    
    # Check if any similar products were found
    if not recommended_products.empty:
        print("Similar Products:\n")
        for product_name in recommended_products:
            print(product_name)
    else:
        print("No similar products found.")
except KeyError:
    print("The entered product name does not exist in the dataset. Please try again.")


