# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings
from scipy.spatial.distance import cosine

# Suppress Warnings
warnings.filterwarnings("ignore")

# Initialize NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Global Variables
stop = set(stopwords.words('english'))
specific_stop_words = {"rs", "flipkart", "buy", "com", "free", "day", "cash", "replacement", "guarantee", "genuine", 
                       "key", "feature", "delivery", "products", "product", "shipping", "online", "india", "stop"}

# Load the Dataset
data = pd.read_csv("flipkart_com-ecommerce_sample.csv")
data["description"] = data["description"].fillna("")

# Text Preprocessing
def remove_stopwords(text):
    """Remove punctuation, stopwords, and specific domain-related words from the text."""
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word not in stop and word not in specific_stop_words]
    return " ".join(tokens)

data["description"] = data["description"].apply(remove_stopwords)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
T_vec_matrix = vectorizer.fit_transform(data["description"])

# Reverse mapping of product names to indices
product_index = pd.Series(data.index, index=data["product_name"]).drop_duplicates()

# Word2Vec Model Loading
filename = r"E:\\Notes all and videos\\NLP By Suven Consultant\\2- Enhansing Recommandation System by Flipkart data\\GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=50000)

# Get Word Embeddings
def get_embedding(word):
    """Fetch the embedding for a word, or return a zero vector if not found."""
    if word in model:
        return model[word]
    else:
        return np.zeros(300)

# Get Sentence Embedding
def get_sentence_embedding(sentence):
    """Compute the average vector for a sentence."""
    tokens = word_tokenize(sentence)
    vectors = np.array([get_embedding(token) for token in tokens if token in model])
    if len(vectors) == 0:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

# Compute Document Embeddings
out_dict = {}
for desc in data["description"]:
    out_dict[desc] = get_sentence_embedding(desc)

# Compute Similarity
def get_similarity(query_embedding, doc_embedding):
    """Compute cosine similarity between two embeddings."""
    return 1 - cosine(query_embedding, doc_embedding)

# Rank Documents
def rank_documents(query):
    """Rank all documents based on their similarity to the query."""
    query_embedding = get_sentence_embedding(query)
    ranked = []
    for desc, doc_embedding in out_dict.items():
        similarity = get_similarity(query_embedding, doc_embedding)
        ranked.append((desc, similarity))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    result = pd.DataFrame(ranked, columns=["description", "similarity"])
    result = pd.merge(data, result, on="description")
    result = result[["product_name", "description", "similarity"]].sort_values(by="similarity", ascending=False)
    return result.head(10)

# User Input and Recommendations
query = input("What would you like to search? ")
print(rank_documents(query))
