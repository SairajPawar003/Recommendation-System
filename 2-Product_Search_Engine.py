#Data Manipulating
import pandas as pd
import numpy as np 

#Data Visualization 
import matplotlib.pyplot as plt 
import scipy.spatial
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
from gensim.models import Word2Vec,KeyedVectors

import itertools 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

#remove warning 
import warnings 
warnings.filterwarnings(action='ignore')


# def predict_products(text):
#     # Getting the index of the input product
#     index = product_index[text]
    
#     # Pairwise similarity scores
#     score_matrix = linear_kernel(T_vec_matrix[index], T_vec_matrix)
#     matching_sc = list(enumerate(score_matrix[0]))
    
#     # Sort the products based on the similarity score
#     matching_sc = sorted(matching_sc, key=lambda x: x[1], reverse=True)
    
#     # Getting the scores of the top 10 most similar products (excluding the input product itself)
#     matching_sc = matching_sc[1:11]  # Fix: Correct slicing
    
#     # Getting product indices of the top matches
#     product_indices = [i[0] for i in matching_sc]
    
#     # Show the similar products
#     return data['product_name'].iloc[product_indices]


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


def remove_stopwords(text, is_lower_case=False):
    pattern=r'[^a-zA-z0-9\s]'
    text = re.sub(pattern,'',text[0])
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stop]
    else :
        filtered_tokens = [token for token in tokens if token.lower() not in stop]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

#obtain the embading use 300 
# def get_embedding(word):
#     if word in model.wv.vocab:
#         return model[word]
#     else:
#         return np.zeros(300)
def get_embedding(word):
    if word in model.key_to_index:
        return model[word]  # Should always be 1-D
    else:
        return np.zeros(model.vector_size)  # Ensure dimensionality matches

    

#Get similarity between the query and document 
# def get_sem(query_embedding,average_vector_doc):
#     sim = [(1-scipy.spatial.distance.cosine(query_embedding,average_vector_doc))]
#     return sim 
# Function to calculate similarity
from scipy.spatial.distance import cosine
def get_sem(query_embedding, doc_embedding):
    if query_embedding.ndim != 1 or doc_embedding.ndim != 1:
        raise ValueError("Both embeddings must be 1-D.")
    if np.all(query_embedding == 0) or np.all(doc_embedding == 0):
        return 0.0
    return 1 - cosine(query_embedding, doc_embedding)

# Rank documents based on query
def Ranked_documents(query):
    # Tokenize query and get embeddings
    query_words = [get_embedding(token) for token in nltk.word_tokenize(query.lower())]
    if not query_words:
        print("Query resulted in no valid tokens. Returning empty results.")
        return pd.DataFrame(columns=['product_name', 'description', 'score'])

    # Compute query embedding (ensure 1-D)
    query_embedding = np.mean(query_words, axis=0)
    if query_embedding.ndim != 1:
        raise ValueError("Query embedding is not 1-D.")

    # Compute document rankings
    rank = [
        (k, get_sem(query_embedding, v)) for k, v in out_dict.items() if v.ndim == 1
    ]

    # # Sort by similarity and return top results
    # rank = sorted(rank, key=lambda t: t[1], reverse=True)
    # rank_df = pd.DataFrame(rank, columns=['Desc', 'score'])
    # result = pd.merge(data1, rank_df, left_on='description', right_on='Desc', how='inner')
    # return result[['product_name', 'description', 'score']]
    # Sort by similarity score in descending order
    rank = sorted(rank, key=lambda t: t[1], reverse=True)
    
    # Convert the sorted list to a DataFrame
    rank_df = pd.DataFrame(rank, columns=['Desc', 'score'])
    
    # Merge data1 with rank_df on the 'description' column
    result = pd.merge(data1, rank_df, left_on='description', right_on='Desc', how='inner')
     # Sort the merged result by 'score' in descending order
    result = result.sort_values(by='score', ascending=False)
    
    # Return the desired subset of columns
    return result[['product_name', 'description', 'score']]


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
'''input_product = input("Enter a product name: ")

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
'''
# *****PRODUCT SEARCH ENGINE************
#Creating list containing description of each product as sublist 
fin=[]
for i in range(len(data['description'])):
    temp=[]
    temp.append(data['description'][i])
    fin = fin + temp
data1 = data[['product_name','description']]
#import thr word2vec
# from gensim.models import Word2Vec,KeyedVectors
filename = r"E:\\Notes all and videos\\NLP By Suven Consultant\\2- Enhansing Recommandation System by Flipkart data\\GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(filename, binary=True,limit=50000)



##preprocesing function used def remove_stopwards
##def get embedding 
#obtaining the average vector for all documents
out_dict={}
for sen in fin : 
    average_vector = (np.mean(np.array([get_embedding(x) for x in nltk.word_tokenize(remove_stopwords(sen))]),axis=0))
    dict = {sen : (average_vector)}
    out_dict.update(dict)

#Get similarity between object and the document 
#def get_sim 
#rank all documents based on similarity 
#def Ranked_document
#call the ir function with the query 
query = input("what would you like to search ")
print(Ranked_documents(query))
