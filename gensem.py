import os
from gensim.models import KeyedVectors

filename = r"E:\\Notes all and videos\\NLP By Suven Consultant\\2- Enhansing Recommandation System by Flipkart data\\GoogleNews-vectors-negative300.bin"
print("File exists:", os.path.exists(filename))
print("Readable:", os.access(filename, os.R_OK))

# try:
#     model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=50000)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error: {e}")

with open(filename, 'rb') as f:
    header = f.read(100)
    print(header)