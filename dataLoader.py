import logging
import numpy as np
import scipy.sparse as sp
import time
import utils
from numpy.linalg import inv

#from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from gensim import corpora, similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.similarities import Similarity
from tqdm import tqdm

from doc_db import DocDB

'''
start_time = time.time()
tfidf_path = "/work/04549/mustaf/maverick/data/TREC/TREC8/TREC8-tfidf-ngram=1-hash=16777216-tokenizer=simple.npz"
matrix, metadata = utils.load_sparse_csr(tfidf_path)
doc_mat = matrix
ngrams = metadata['ngram']
hash_size = metadata['hash_size']
#tokenizer = tokenizers.get_class(metadata['tokenizer'])()
doc_freqs = metadata['doc_freqs'].squeeze()
doc_dict = metadata['doc_dict']
num_docs = len(doc_dict[0])
print "loading take :", time.time() - start_time


def get_doc_index(doc_id):
    """Convert doc_id --> doc_index"""
    return doc_dict[0][doc_id]


def get_doc_id(doc_index):
    """Convert doc_index --> doc_id"""
    return doc_dict[1][doc_index]


print doc_mat.shape
train_index = [2,4,5]
X = doc_mat[:,train_index]
X = X.todense()
Xinv = inv(X)
y = [1,0,1]

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(Xinv,y)

print clf.predict(Xinv)

'''

PROCESS_DB = DocDB('/work/04549/mustaf/maverick/data/TREC/TREC8/TREC8normalized.db')
doc_ids = PROCESS_DB.get_doc_ids()
DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

def stream_db(dictionary_use=True):
    global doc_ids

    with tqdm(total=len(doc_ids)) as pbar:
        for doc_id in doc_ids:
            #print c, doc_id
            if dictionary_use == True:
                yield fetch_text(doc_id).split()
            else:
                yield dictionary.doc2bow(fetch_text(doc_id).split())

            pbar.update()


def stream_corpus(filename, dictionary_use=True):
    for line in open(filename):
        # assume there's one document per line, tokens separated by whitespace
        if dictionary_use == True:
            yield line.lower().split()
        else:
            yield dictionary.doc2bow(line.lower().split())

print ("Started Dictionary")
start = time.time()
#dictionary = Dictionary(stream_corpus("mycorpus.txt"),prune_at = None)
dictionary = Dictionary(stream_db(),prune_at = None)

print "complete", time.time() - start


#for vector in stream_db( False):  # load one vector into memory at a time
#    print(vector)

