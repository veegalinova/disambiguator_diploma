import string
import json
from glob import glob

import nmslib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from yargy.tokenizer import MorphTokenizer
from razdel import sentenize

from parser import TextProcessor

morph_tokenizer = MorphTokenizer()

with open("config.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)

stop_words = stopwords.words('russian') + [s for s in string.punctuation]


def tokenizer(s):
    tokens = morph_tokenizer(s)
    return [token.normalized for token in list(tokens) if token not in stop_words]

files = glob('text_data/*/*.htm')

parser = TextProcessor(config['database'])
data = []
for file in files:
    text = parser.extract_text_from_htm(file)
    data.extend([sent.text for sent in list(sentenize(text))])

print(data)
vectorizer = CountVectorizer(stop_words=stop_words, tokenizer=tokenizer)
text = vectorizer.fit_transform(data)
print(text.shape)

M = 30
efC = 100

num_threads = 4
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}

index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
index.addDataPointBatch(text)
index.createIndex(index_time_params)

neighbours = index.knnQueryBatch(text, k=3, num_threads=4)
print(neighbours)

desc = []
for _ in neighbours:
  d = []
  prev = -1
  for j in zip(_[0], _[1]):
    if j[1] != prev:
      d.append(data[j[0]])
      prev = j[1]
  desc.append(d)

with open('res2.txt', 'w') as file:
    json.dump(desc, file)