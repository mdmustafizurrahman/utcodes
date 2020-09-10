#!/usr/bin/env python3
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import re
from nltk.corpus import stopwords
import time
import pickle
import scipy.sparse as sp
import copy
#import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from gensim import corpora, similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.similarities import Similarity

from global_definition import *
from topic_description import *

'''
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
'''

# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------

def init(filename):
    #global PREPROCESS_FN
    #if filename:
    #    PREPROCESS_FN = import_module(filename).preprocess
    pass

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in sorted(filenames):
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))



def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []
    with open(filename) as f:
        docId = filename[filename.rfind('/')+1:filename.index('.')]

        # counting the line number until '---Terms---'
        count = 0
        for lines in f:
            if lines.find("Terms") > 0:
                count = count + 1
                break
            count = count + 1

        # skipping the lines until  '---Terms---' and reading the rest
        c = 0
        tmpStr = ""
        # print "count:", count
        # f = open(path)
        for lines in f:
            if c < count:
                c = c + 1
                continue
            values = lines.split()
            c = c + 1
            # print values[0], values[1], values[2]
            tmpStr = tmpStr + " " + str(values[2])

        documents.append((docId, review_to_words(tmpStr)))
    return documents # returning a list of documents # but documents only contain one document



def stream_corpus(data_path, dictionary, files, num_workers=None):
    workers = ProcessPool(num_workers)
    #files = [f for f in iter_files(data_path)]

    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            yield dictionary.doc2bow(pairs[0][1].split())  # pairs[0][0]-->docId, pairs[0][1]-->documentContent
            pbar.update()


def store_contents(data_path, save_path, datasource, processOnlyFilesinOriginalQrels, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    print save_path
    print data_path
    docIds = [] # list of TREC DocID
    docIdToDocIndex = {} # key is DocID, value is docIndex
    docIndex = 0

    workers = ProcessPool(num_workers)
    files = []
    if processOnlyFilesinOriginalQrels == True:
        topicData = TRECTopics(datasource, start_topic[datasource], end_topic[datasource])
        qrelDocList = topicData.qrelDocIdLister(qrelAddress[datasource], save_path, topic_original_qrels_doc_list_file_name)
        files = []
        for docId in qrelDocList:
            fileid = docId+'.txt'
            files.append(os.path.join(data_path,fileid))
        #files = [f for f in iter_files(data_path) if os.path.splitext(os.path.basename(f))[0] in qrelDocList]
        print "Number of unique documents in the qrels", len(files)

    else:
        files = [f for f in iter_files(data_path)]

    dictionary = Dictionary()
    count = 0

    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(pairs)
            dictionary.add_documents([pairs[0][1].split()]) # pairs[0][0]-->docId, pairs[0][1]-->documentContent
            docIdToDocIndex[pairs[0][0]] = docIndex
            docIds.append(pairs[0][0])
            docIndex = docIndex + 1
            pbar.update()

    print ("Number of documents:", docIndex, len(docIds), len(docIdToDocIndex))
    total_documents = len(docIds)
    metadata = {}
    metadata['docIdToDocIndex'] = docIdToDocIndex
    metadata['docIndexToDocId'] = docIds
    # protocol 2 for version compaitability
    pickle.dump(metadata, open(save_path + meta_data_file_name[datasource], 'wb'), protocol=2)

    # keep only words that
    # exist within at least 20 articles
    # keep only the top most freqent 15000 tokens
    dictionary.filter_extremes(no_below=20, keep_n=dictionary_features_number)
    dictionary.compactify()
    dictionary.save_as_text(save_path + dictionary_name)


    dictionary = Dictionary.load_from_text(save_path + dictionary_name)
    start_time = time.time()
    corpus_bow_stream = stream_corpus(data_path, dictionary, files)
    MmCorpus.serialize(save_path + corpus_bow_file_name, corpus_bow_stream, progress_cnt=10000)
    corpus_bow = MmCorpus(save_path + corpus_bow_file_name)
    model_tfidf = TfidfModel(corpus_bow, id2word=dictionary, normalize=True)
    model_tfidf.save(save_path + corpus_tfidf_model_file_name)
    corpus_tfidf = model_tfidf[corpus_bow]  # apply model
    MmCorpus.serialize(save_path + corpus_tfidf_file_name, corpus_tfidf, progress_cnt=1000)

    # Load the tf-idf corpus back from disk.
    corpus_tfidf = MmCorpus(save_path + corpus_tfidf_file_name)
    #n_items = len(dictionary)
    #print corpus_tfidf

    # CSR matrix construction phase
    indptr = [0]
    indices = []
    data = []
    # processing took 9:26s
    with tqdm(total=total_documents) as pbar:
        for doc in corpus_tfidf:
            for (index, values) in doc:
                indices.append(index)
                data.append(values)
            indptr.append(len(indices))
            pbar.update()

    start = time.time()
    sparse_matrix = sp.csr_matrix((data, indices, indptr), dtype=float)
    # saving took 01:21s
    sp.save_npz(save_path + csr_matrix_file_name[datasource], sparse_matrix)
    print "Finished in:", (time.time() - start)

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------

# run command
# python build_dictionary.py /work/04549/mustaf/maverick/data/TREC/TREC8/vectorizedTREC/ /work/04549/mustaf/maverick/data/TREC/TREC8/sparseTREC/

# python build_dictionary.py /work/04549/mustaf/maverick/data/TREC/TREC8/vectorizedTREC/ /work/04549/mustaf/maverick/data/TREC/TREC8/sparseTRECqrels/ TREC8 --processqrelsonly True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save_path = /export/home/u16/nahid/data/sqliteDB/
    # data_path = /export/home/u16/nahid/data/vectorizedTREC/TREC8/

    # data_path = /work/04549/mustaf/maverick/data/TREC/TREC8/vectorizedTREC/
    # save_path = /work/04549/mustaf/maverick/data/TREC/TREC8/sparseTREC/
    # save_path = /work/04549/mustaf/maverick/data/TREC/TREC8/sparseTRECqrels/
    parser.add_argument('data_path', type=str, help='/path/to/data/')
    parser.add_argument('save_path', type=str, help='/path/to/saved/')
    parser.add_argument('datasource', type=str, help='name of the datasource e,g. TREC8, TREC7')
    parser.add_argument('--processqrelsonly', type=bool, default=False,
                        help=('if the whole collection is only the set of documents in the qrels set it to True'))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    store_contents(
        args.data_path, args.save_path, args.datasource, args.processqrelsonly, args.num_workers
    )

'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
indptr = [0]
indices = []
data = []
vocabulary = {}
for d in docs:
     for term in d:
         index = vocabulary.setdefault(term, len(vocabulary))
         indices.append(index)
         data.append(1)
     indptr.append(len(indices))

csr_matrix((data, indices, indptr), dtype=int).toarray()
array([[2, 1, 0, 0],
       [0, 1, 1, 1]])

'''