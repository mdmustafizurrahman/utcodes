'''Active learning for labeling the relevant document for TREC-8 dataset
@author: Md Mustafizur Rahman (nahid@utexas.edu)'''

import os
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
import math
import nltk
import copy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
'''
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
'''
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import collections
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
import pickle
from math import log
import pandas as pd
import Queue

import logging
logging.basicConfig()

TEXT_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/IndriData/'
RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/relevance.txt'
docrepresentation = "TF-IDF"  # can be BOW, TF-IDF

# Fixed parameter set for JASIST submission
sampling=False # can be True or False, not needed in initial phase as we are making sure that there
# is 5 and 5 non-rel and rel docs
command_prompt_use = True
correction = 'False' # not used anymore but keep it fixed at False in JASIST paper
train_per_centage_flag = 'True' # This is also fixed true in JASIST paper

base_address = "/work/04549/mustaf/maverick/data/TREC/deterministic1/"


#if command_prompt_use == True:
'''
datasource = sys.argv[1] # can be  dataset = ['TREC8', 'gov2', 'WT']
protocol = sys.argv[2]
use_ranker = sys.argv[3]
iter_sampling = sys.argv[4]
correction = sys.argv[5] #'SAL' can be ['SAL', 'CAL', 'SPL']
train_per_centage_flag = sys.argv[6]
'''

#parameter set # all FLAGS must be string
'''
datasource = 'TREC8'  # can be  dataset = ['TREC8', 'gov2', 'WT']
protocol = 'CAL'  # 'SAL' can be ['SAL', 'CAL', 'SPL']
use_ranker = 'True' # false means IS, true means RDS
iter_sampling = 'True' # sampling in the active iteration phase
'''

datasource = sys.argv[1]  # can be  dataset = ['TREC8', 'gov2', 'WT']
protocol = sys.argv[2]  # 'SAL' can be ['SAL', 'CAL', 'SPL']
use_ranker = sys.argv[3] # false means IS, true means RDS
iter_sampling = sys.argv[4] # sampling in the active iteration phase


print "Ranker_use", use_ranker
print "iter_sampling", iter_sampling
print "correction", correction
print "train_percenetae", train_per_centage_flag


test_size = 0    # the percentage of samples in the dataset that will be
#test_size_set = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
test_size_set = [0.2]
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

ranker_location = {}
ranker_location["WT2013"] = "/work/04549/mustaf/maverick/data/TREC/WT2013/input.ICTNET13RSR2"
ranker_location["WT2014"] = "/work/04549/mustaf/maverick/data/TREC/WT2014/input.Protoss"
ranker_location["gov2"] = "/work/04549/mustaf/maverick/data/TREC/gov2/input.indri06AdmD"
ranker_location["TREC8"] = "/work/04549/mustaf/maverick/data/TREC/TREC8/input.ibmg99b"

n_labeled =  10 #50      # number of samples that are initially labeled
batch_size = 25 #50
preloaded = True
topicSkipList = [202,225,255,278,805] # For IS seed set
# for RDS seed set
#topicSkipList = [202,210,225,234,235,238,244,251,255,262,269,271,278,283,289,291,803,805]
skipList = []
topicBucketList = []
processed_file_location = ''
start_topic = 0
end_topic = 0

base_address = base_address +str(datasource)+"/"
if use_ranker == 'True':
    base_address = base_address + "ranker/"
    use_ranker = True
else:
    base_address = base_address + "no_ranker/"
    use_ranker = False
if iter_sampling == 'True':
    base_address = base_address + "oversample/"
    iter_sampling = True
if iter_sampling == 'False':
    iter_sampling = False
    base_address = base_address + "no_oversample/"
if correction == 'True':
    base_address = base_address + "htcorrection/"
    correction = True
if correction == 'False':
    correction = False

if train_per_centage_flag == 'True':
    train_per_centage_flag = True
else:
    train_per_centage_flag = False

print "base address:", base_address


if iter_sampling == True and correction == True:
    print "Over sampling and HT correction cannot be done together"
    exit(-1)

#if iter_sampling == False and correction == False:
#    print "Over sampling and HT correction cannot be both false together"
#    exit(-1)

print "base address:", base_address
if datasource=='TREC8':
    processed_file_location = '/work/04549/mustaf/maverick/data/TREC/TREC8/processed.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/TREC8/relevance.txt'
    start_topic = 401
    end_topic = 451
    datasetsize = 86830
elif datasource=='gov2':
    processed_file_location = '/work/04549/mustaf/maverick/data/TREC/gov2/processed.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/gov2/qrels.tb06.top50.txt'
    start_topic = 801
    end_topic = 851
    datasetsize = 31984
elif datasource=='WT2013':
    processed_file_location = '/work/04549/mustaf/maverick/data/TREC/WT2013/processed_new.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/WT2013/qrelsadhoc2013.txt'
    start_topic = 201
    end_topic = 251
    #datasetsize = 14474 #contains 13265 docs after skipping the topic in skip list
    datasetsize = 13265
else:
    processed_file_location = '/work/04549/mustaf/maverick/data/TREC/WT2014/processed_new.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/WT2014/qrelsadhoc2014.txt'
    start_topic = 251
    end_topic = 301
    datasetsize = 13483
    #datasetsize = 14432 # contains 13483 docs after skipping the topic in skip list



#print result_location
#exit(0)
class relevance(object):
    def __init__(self, priority, index):
        self.priority = priority
        self.index = index
        return
    def __cmp__(self, other):
        return -cmp(self.priority, other.priority)

def calculate_entropy(a,b):
    first_part = 0.0
    second_part = 0.0
    if a>0:
        first_part = a * log(a, 2)
    if b>0:
        second_part = b* log(b, 2)

    return (-1)*(first_part+second_part)

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


def run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size):
    E_in, E_out = [], []

    for _ in range(quota):


        # Standard usage of libact objects
        ask_id = qs.make_query()
        #print  ask_id
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)


        model.train(trn_ds)
        #model.predict(tst_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out, model




all_reviews = {}
learning_curve = {} # per batch value for  validation set

if preloaded==False:
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        #print name
        #if name == "ft":
        path = os.path.join(TEXT_DATA_DIR, name)
        print path

        f = open(path)
        docNo = name[0:name.index('.')]
        #print docNo

        # counting the line number until '---Terms---'
        count = 0
        for lines in f:
            if lines.find("Terms")>0:
                count = count + 1
                break
            count = count + 1

        # skipping the lines until  '---Terms---' and reading the rest
        c = 0
        tmpStr = ""
        #print "count:", count
        #f = open(path)
        for lines in f:
            if c < count:
                c = c + 1
                continue
            values = lines.split()
            c = c + 1
            #print values[0], values[1], values[2]
            tmpStr = tmpStr + " "+ str(values[2])
        print tmpStr
        #exit(0)

        #if docNo in docNo_label:
        all_reviews[docNo] = (review_to_words(tmpStr))

        f.close()

    output = open(processed_file_location, 'ab+')
    # data = {'a': [1, 2, 3], }

    pickle.dump(all_reviews, output)
    output.close()

else:
    input = open(processed_file_location, 'rb')
    all_reviews = pickle.load(input)
    print "pickle loaded"


print('Reading the Ranker label Information')
f = open(ranker_location[datasource])
print "Ranker:", f
tmplist = []
Ranker_topic_to_doclist = {}
for lines in f:
    values = lines.split()
    topicNo = values[0]
    docNo = values[2]
    if (Ranker_topic_to_doclist.has_key(topicNo)):
        tmplist.append(docNo)
        Ranker_topic_to_doclist[topicNo] = tmplist
    else:
        tmplist = []
        tmplist.append(docNo)
        Ranker_topic_to_doclist[topicNo] = tmplist
f.close()
# print len(topic_to_doclist)

for test_size in test_size_set:
    seed = 1335
    for fold in xrange(1,2):
        np.random.seed(seed)
        seed = seed + fold
        result_location = base_address + 'result_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location_base = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_'
        human_label_location = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_'

        learning_curve_location = base_address + 'learning_curve_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        s = "";
        pred_str = ""
        #for topic in sorted(topic_to_doclist.keys()):
        for topic in xrange(start_topic,end_topic):
            print "Topic:", topic
            if topic in topicSkipList:
                print "Skipping Topic :", topic
                continue
            topic = str(topic)

            topic_to_doclist = {}  # key is the topic(string) and value is the list of docNumber
            docNo_label = {}  # key is the DocNo and the value is the label
            docIndex_DocNo = {} # key is the index used in my code value is the actual DocNo
            docNo_docIndex = {} # key is the DocNo and the value is the index assigned by my code
            best_f1 = 0.0  # best f1 considering per iteraton of active learning
            print('Reading the relevance label')
            # file open
            f = open(RELEVANCE_DATA_DIR)
            print f
            tmplist = []
            #g = 0
            for lines in f:
                #print lines
                #g = g + 1
                #if g>2739:
                #    break
                values = lines.split()
                topicNo = values[0]
                if topicNo != topic:
                    #print "Skipping", topic, topicNo
                    continue
                docNo = values[2]
                label = int(values[3])
                if label > 1:
                    label = 1
                if label < 0:
                    label = 0
                docNo_label[docNo] = label
                if (topic_to_doclist.has_key(topicNo)):
                    tmplist.append(docNo)
                    topic_to_doclist[topicNo] = tmplist
                else:
                    tmplist = []
                    tmplist.append(docNo)
                    topic_to_doclist[topicNo] = tmplist
            f.close()
            #print len(topic_to_doclist)
            docList = topic_to_doclist[topic]
            print 'number of documents', len(docList)
            #print docList
            #print ('Processing news text for topic number')
            relevance_label = []
            judged_review = []

            docIndex = 0
            for documentNo in docList:
                if all_reviews.has_key(documentNo):
                    #print "in List", documentNo
                    #print documentNo, 'len:', type(all_reviews[documentNo])

                    #print all_reviews[documentNo]
                    #exit(0)
                    docIndex_DocNo[docIndex] = documentNo
                    docNo_docIndex[documentNo] = docIndex
                    docIndex = docIndex + 1
                    judged_review.append(all_reviews[documentNo])
                    relevance_label.append(docNo_label[documentNo])


            if docrepresentation == "TF-IDF":
                print "Using TF-IDF"
                vectorizer = TfidfVectorizer( analyzer = "word",   \
                                         tokenizer = None,    \
                                         preprocessor = None, \
                                         stop_words = None,   \
                                         max_features = 15000)

                bag_of_word = vectorizer.fit_transform(judged_review)


            elif docrepresentation == "BOW":
                # Initialize the "CountVectorizer" object, which is scikit-learn's
                # bag of words tool.
                print "Uisng Bag of Word"
                vectorizer = CountVectorizer(analyzer = "word",   \
                                             tokenizer = None,    \
                                             preprocessor = None, \
                                             stop_words = None,   \
                                             max_features = 15000)

                # fit_transform() does two functions: First, it fits the model
                # and learns the vocabulary; second, it transforms our training data
                # into feature vectors. The input to fit_transform should be a list of
                # strings.
                bag_of_word = vectorizer.fit_transform(judged_review)

            # Numpy arrays are easy to work with, so convert the result to an
            # array
            bag_of_word = bag_of_word.toarray()
            print bag_of_word.shape
            #vocab = vectorizer.get_feature_names()
            #print vocab

            # Sum up the counts of each vocabulary word
            #dist = np.sum(bag_of_word, axis=0)

            # For each, print the vocabulary word and the number of times it
            # appears in the training set
            #for tag, count in zip(vocab, dist):
            #    print count, tag

            print "Bag of word completed"

            X= bag_of_word
            y= relevance_label

            # print len(y)
            # print y
            numberOne = y.count(1)
            # print "Number of One", numberOne

            numberZero = y.count(0)
            print "Number of One", numberOne
            print "Number of Zero", numberZero
            datasize = len(X)
            prevelance = (numberOne * 1.0) / datasize
            print "=========Before Sampling======"

            print "Whole Dataset size: ", datasize
            print "Number of Relevant", numberOne
            print "Number of non-relevant", numberZero
            print "prevelance ratio", prevelance * 100

            #print "After", y_train


            print '----Started Training----'
            model = LogisticRegression(C=100000000)
            size = len(X) - n_labeled

            if size<0:
                print "Train Size:", len(X) , "seed:", n_labeled
                size = len(X)

            if use_ranker == True:

                initial_X_train = []
                initial_y_train = []

                train_index_list = []

                # collecting the seed list from the Rankers
                seed_list = Ranker_topic_to_doclist[topic]
                seed_counter = 0
                seed_one_counter = 0
                seed_zero_counter = 0
                ask_for_label = 0
                loopCounter = 0

                seed_size_limit = math.ceil(train_per_centage[loopCounter] * len(X))
                print "Initial Seed Limit", seed_size_limit
                seed_start = 0
                seed_counter = 0
                while True:
                    print "seed size limit:", seed_size_limit
                    while seed_counter < seed_size_limit:
                        documentNumber = seed_list[seed_counter]
                        seed_counter = seed_counter + 1
                        if documentNumber not in docNo_docIndex:
                            continue
                        index = docNo_docIndex[documentNumber]
                        train_index_list.append(index)
                        labelValue = int(docNo_label[documentNumber])
                        ask_for_label = ask_for_label + 1
                        initial_X_train.append(X[index])
                        initial_y_train.append(labelValue)
                        if labelValue == 1:
                            seed_one_counter = seed_one_counter + 1
                        if labelValue == 0:
                            seed_zero_counter = seed_zero_counter + 1

                        #print seed_one_counter, seed_zero_counter
                    if seed_zero_counter == 0 or seed_one_counter == 0:
                        print "Seed Size Limit:", seed_size_limit, "Seed Explored so far:", seed_counter
                        print seed_one_counter, seed_zero_counter
                        ####################################Store Result###################
                        y_pred_all = {}

                        human_label_str = ""

                        for train_index in train_index_list:
                            y_pred_all[train_index] = y[train_index]
                            docNo = docIndex_DocNo[train_index]
                            human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[train_index]) + "\n"
                        human_label_location_final = human_label_location + str(train_per_centage[loopCounter]) + '_human_.txt'
                        text_file = open(human_label_location_final, "a")
                        text_file.write(human_label_str)
                        text_file.close()

                        for train_index in xrange(0, len(X)):
                            if train_index not in train_index_list:
                                y_pred_all[train_index] = 0 # consider all are non-relevants

                        y_pred = []
                        for key, value in y_pred_all.iteritems():
                            # print (key,value)
                            y_pred.append(value)

                        ##################

                        pred_topic_str = ""
                        for docIndex in xrange(0, len(X)):
                            docNo = docIndex_DocNo[docIndex]
                            pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"

                        predicted_location_final = predicted_location_base + str(
                            train_per_centage[loopCounter]) + '.txt'
                        text_file = open(predicted_location_final, "a")
                        text_file.write(pred_topic_str)
                        text_file.close()

                        f1score = f1_score(y, y_pred, average='binary')

                        if (learning_curve.has_key(train_per_centage[loopCounter])):
                            tmplist = learning_curve.get(train_per_centage[loopCounter])
                            tmplist.append(f1score)
                            learning_curve[train_per_centage[loopCounter]] = tmplist
                        else:
                            tmplist = []
                            tmplist.append(f1score)
                            learning_curve[train_per_centage[loopCounter]] = tmplist

                        #learning_batch_size = learning_batch_size + batch_size
                        precision = precision_score(y, y_pred, average='binary')
                        recall = recall_score(y, y_pred, average='binary')

                        print "Score in non-active stage"
                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score


                        loopCounter = loopCounter + 1
                        seed_size_limit = math.ceil(train_per_centage[loopCounter] * len(X))
                    if seed_zero_counter > 0 and seed_one_counter > 0:
                        print "Requirement made at counter", loopCounter,  "(1:)", seed_one_counter, "(0):", seed_zero_counter
                        print "Seed Size Limit:", seed_size_limit, "Seed Explored so far:", seed_counter
                        break

                unmodified_train_X = copy.deepcopy(initial_X_train)
                unmodified_train_y = copy.deepcopy(initial_y_train)
                sampling_weight = []

                for sampling_index in xrange(0, len(initial_X_train)):
                    sampling_weight.append(1.0)

                # Ranker needs oversampling, but when HTCorrection true we cannot perform oversample
                if use_ranker == True and correction == False:
                    print "Oversampling in the seed list"
                    ros = RandomOverSampler()
                    # ros = RandomUnderSampler()
                    # ros = SMOTE(random_state=42)
                    # ros = ADASYN()
                    initial_X_train_sampled, initial_y_train_sampled = ros.fit_sample(initial_X_train, initial_y_train)
                    initial_X_train = initial_X_train_sampled
                    initial_y_train = initial_y_train_sampled

                    initial_X_train = initial_X_train.tolist()
                    initial_y_train = initial_y_train.tolist()

                initial_X_test = []
                initial_y_test = []

                test_index_list = {}
                test_index_counter = 0
                for train_index in xrange(0, len(X)):
                    if train_index not in train_index_list:
                        initial_X_test.append(X[train_index])
                        test_index_list[test_index_counter] = train_index
                        test_index_counter = test_index_counter + 1
                        initial_y_test.append(y[train_index])

                # initial_X_test = X_train[n_labeled:]
                # initial_y_test = y_train[n_labeled:]

                # initial_X_test = initial_X_test.tolist()
                # initial_y_test = initial_y_test.tolist()

                print "Before Loop Lenght:", len(initial_X_train), len(initial_y_train)
                predictableSize = len(initial_X_test)
                isPredictable = [1] * predictableSize  # initially we will predict all

                #loopCounter = 0
                best_model = 0
                learning_batch_size = n_labeled  # starts with the seed size

                # Train_per_centage_flag is True is used for JASIST submission
                if train_per_centage_flag == False:
                    numberofloop = math.ceil(size / batch_size)
                    if numberofloop == 0:
                        numberofloop = 1
                    print "Number of loop", numberofloop

                    while loopCounter <= numberofloop:
                        print "Loop:", loopCounter

                        loopDocList = []

                        if protocol == 'SPL':
                            model = LogisticRegression(C=100000000)

                        print len(initial_X_train), len(sampling_weight)
                        model.fit(initial_X_train, initial_y_train, sample_weight=sampling_weight)

                        y_pred_all = {}

                        for train_index in train_index_list:
                            y_pred_all[train_index] = y[train_index]

                        for train_index in xrange(0, len(X)):
                            if train_index not in train_index_list:
                                y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

                        y_pred = []
                        for key, value in y_pred_all.iteritems():
                            # print (key,value)
                            y_pred.append(value)

                        f1score = f1_score(y, y_pred, average='binary')

                        if (learning_curve.has_key(learning_batch_size)):
                            tmplist = learning_curve.get(learning_batch_size)
                            tmplist.append(f1score)
                            learning_curve[learning_batch_size] = tmplist
                        else:
                            tmplist = []
                            tmplist.append(f1score)
                            learning_curve[learning_batch_size] = tmplist

                        learning_batch_size = learning_batch_size + batch_size
                        precision = precision_score(y, y_pred, average='binary')
                        recall = recall_score(y, y_pred, average='binary')

                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        if isPredictable.count(1) == 0:
                            break

                        # if f1score == 1.0:
                        #    break
                        #    print "BREAKING LOOP BECAUSE F-1 is 1.0"

                        # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
                        '''
                        precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

                        recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

                        So now you are dividing 0/0.'''
                        # here is queueSize is the number of predictable element
                        queueSize = isPredictable.count(1)

                        if protocol == 'CAL':
                            print "####CAL####"
                            queue = Queue.PriorityQueue(queueSize)
                            y_prob = []
                            counter = 0
                            sumForCorrection = 0.0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    # reshapping reshape(1,-1) because it does not take one emelemt array
                                    # list does not contain reshape so we are using np,array
                                    # model.predit returns two value in index [0] of the list
                                    y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))[0]
                                    # y_prob = model.predict(initial_X_test[counter])
                                    # print y_prob
                                    queue.put(relevance(y_prob[1], counter))
                                    sumForCorrection = sumForCorrection + y_prob[1]

                            batch_counter = 0
                            while not queue.empty():
                                if batch_counter == batch_size:
                                    break
                                item = queue.get()
                                # print len(item)
                                # print item.priority, item.index

                                isPredictable[item.index] = 0  # not predictable
                                # initial_X_train.append(initial_X_test[item.index])
                                # initial_y_train.append(initial_y_test[item.index])

                                if correction == True:
                                    correctionWeight = item.priority / sumForCorrection
                                    # correctedItem = [x / correctionWeight for x in initial_X_test[item.index]]
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(correctionWeight)
                                else:
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(1.0)
                                unmodified_train_y.append(initial_y_test[item.index])

                                train_index_list.append(test_index_list[item.index])

                                # print "Docs:", initial_X_test[item.index]
                                loopDocList.append(int(initial_y_test[item.index]))
                                batch_counter = batch_counter + 1
                                # print X_train.append(X_test.pop(item.priority))

                        if protocol == 'SAL':
                            print "####SAL####"
                            queue = Queue.PriorityQueue(queueSize)
                            y_prob = []
                            counter = 0
                            sumForCorrection = 0.0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    # reshapping reshape(1,-1) because it does not take one emelemt array
                                    # list does not contain reshape so we are using np,array
                                    # model.predit returns two value in index [0] of the list
                                    y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))[0]
                                    #entropy = (-1) * (y_prob[0] * log(y_prob[0], 2) + y_prob[1] * log(y_prob[1], 2))
                                    entropy = calculate_entropy(y_prob[0], y_prob[1])
                                    queue.put(relevance(entropy, counter))
                                    sumForCorrection = sumForCorrection + entropy

                            batch_counter = 0
                            while not queue.empty():
                                if batch_counter == batch_size:
                                    break
                                item = queue.get()
                                # print len(item)
                                # print item.priority, item.index
                                isPredictable[item.index] = 0  # not predictable
                                if correction == True:
                                    correctionWeight = item.priority / sumForCorrection
                                    # correctedList = [x / correctionWeight for x in initial_X_test[item.index]]
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(correctionWeight)
                                else:
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(1.0)

                                unmodified_train_y.append(initial_y_test[item.index])
                                train_index_list.append(test_index_list[item.index])

                                loopDocList.append(int(initial_y_test[item.index]))
                                batch_counter = batch_counter + 1
                                # print X_train.append(X_test.pop(item.priority))

                        if protocol == 'SPL':
                            print "####SPL####"
                            randomArray = []
                            randomArrayIndex = 0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    randomArray.append(counter)
                                    randomArrayIndex = randomArrayIndex + 1
                            import random

                            random.shuffle(randomArray)

                            batch_counter = 0
                            for batch_counter in xrange(0, batch_size):
                                if batch_counter > len(randomArray) - 1:
                                    break
                                itemIndex = randomArray[batch_counter]
                                isPredictable[itemIndex] = 0
                                unmodified_train_X.append(initial_X_test[itemIndex])
                                unmodified_train_y.append(initial_y_test[itemIndex])
                                sampling_weight.append(1.0)
                                train_index_list.append(test_index_list[itemIndex])
                                loopDocList.append(int(initial_y_test[itemIndex]))

                        initial_X_train[:] = []
                        initial_y_train[:] = []
                        initial_X_train = copy.deepcopy(unmodified_train_X)
                        initial_y_train = copy.deepcopy(unmodified_train_y)
                        if iter_sampling == True:
                            print "Oversampling in the active iteration list"
                            ros = RandomOverSampler()
                            initial_X_train = None
                            initial_y_train = None
                            initial_X_train, initial_y_train = ros.fit_sample(unmodified_train_X, unmodified_train_y)

                        loopCounter = loopCounter + 1

                else:
                    print "Loop Counter after seed:", loopCounter
                    numberofloop = len(train_per_centage)
                    train_size_controller = len(unmodified_train_X)

                    while loopCounter < numberofloop:
                        size_limit = math.ceil(train_per_centage[loopCounter]*len(X))

                        print "Loop:", loopCounter
                        print "Initial size:",train_size_controller, "limit:", size_limit

                        loopDocList = []

                        if protocol == 'SPL':
                            model = LogisticRegression(C=100000000)

                        print len(initial_X_train)
                        if correction == True:
                            model.fit(initial_X_train, initial_y_train, sample_weight=sampling_weight)
                        else:
                            model.fit(initial_X_train, initial_y_train)

                        y_pred_all = {}

                        human_label_str = ""

                        for train_index in train_index_list:
                            y_pred_all[train_index] = y[train_index]
                            docNo = docIndex_DocNo[train_index]
                            human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[train_index]) + "\n"
                        human_label_location_final = human_label_location + str(train_per_centage[loopCounter]) + '_human_.txt'
                        text_file = open(human_label_location_final, "a")
                        text_file.write(human_label_str)
                        text_file.close()

                        for train_index in xrange(0, len(X)):
                            if train_index not in train_index_list:
                                y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

                        y_pred = []
                        for key, value in y_pred_all.iteritems():
                            # print (key,value)
                            y_pred.append(value)

                        ##################

                        pred_topic_str = ""
                        for docIndex in xrange(0, len(X)):
                            docNo = docIndex_DocNo[docIndex]
                            pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"

                        predicted_location_final = predicted_location_base + str(
                            train_per_centage[loopCounter]) + '.txt'
                        text_file = open(predicted_location_final, "a")
                        text_file.write(pred_topic_str)
                        text_file.close()

                        f1score = f1_score(y, y_pred, average='binary')

                        if (learning_curve.has_key(train_per_centage[loopCounter])):
                            tmplist = learning_curve.get(train_per_centage[loopCounter])
                            tmplist.append(f1score)
                            learning_curve[train_per_centage[loopCounter]] = tmplist
                        else:
                            tmplist = []
                            tmplist.append(f1score)
                            learning_curve[train_per_centage[loopCounter]] = tmplist

                        learning_batch_size = learning_batch_size + batch_size
                        precision = precision_score(y, y_pred, average='binary')
                        recall = recall_score(y, y_pred, average='binary')

                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        if isPredictable.count(1) == 0:
                            break

                        '''
                        precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

                        recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

                        So now you are dividing 0/0.'''
                        # here is queueSize is the number of predictable element
                        queueSize = isPredictable.count(1)

                        if protocol == 'CAL':
                            print "####CAL####"
                            queue = Queue.PriorityQueue(queueSize)
                            y_prob = []
                            counter = 0
                            sumForCorrection = 0.0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    # reshapping reshape(1,-1) because it does not take one emelemt array
                                    # list does not contain reshape so we are using np,array
                                    # model.predit returns two value in index [0] of the list
                                    y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))[0]
                                    # y_prob = model.predict(initial_X_test[counter])
                                    # print y_prob
                                    queue.put(relevance(y_prob[1], counter))
                                    sumForCorrection = sumForCorrection + y_prob[1]

                            batch_counter = 0
                            while not queue.empty():
                                if train_size_controller == size_limit:
                                    break
                                item = queue.get()
                                # print len(item)
                                # print item.priority, item.index

                                isPredictable[item.index] = 0  # not predictable
                                # initial_X_train.append(initial_X_test[item.index])
                                # initial_y_train.append(initial_y_test[item.index])

                                if correction == True:
                                    correctionWeight = item.priority / sumForCorrection
                                    # correctedItem = [x / correctionWeight for x in initial_X_test[item.index]]
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(correctionWeight)
                                else:
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(1.0)
                                unmodified_train_y.append(initial_y_test[item.index])

                                train_index_list.append(test_index_list[item.index])

                                # print "Docs:", initial_X_test[item.index]
                                loopDocList.append(int(initial_y_test[item.index]))
                                train_size_controller = train_size_controller + 1
                                # print X_train.append(X_test.pop(item.priority))

                        if protocol == 'SAL':
                            print "####SAL####"
                            queue = Queue.PriorityQueue(queueSize)
                            y_prob = []
                            counter = 0
                            sumForCorrection = 0.0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    # reshapping reshape(1,-1) because it does not take one emelemt array
                                    # list does not contain reshape so we are using np,array
                                    # model.predit returns two value in index [0] of the list
                                    y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))[0]
                                    #entropy = (-1) * (y_prob[0] * log(y_prob[0], 2) + y_prob[1] * log(y_prob[1], 2))
                                    entropy = calculate_entropy(y_prob[0], y_prob[1])
                                    queue.put(relevance(entropy, counter))
                                    sumForCorrection = sumForCorrection + entropy

                            batch_counter = 0
                            while not queue.empty():
                                if train_size_controller == size_limit:
                                    break
                                item = queue.get()
                                # print len(item)
                                # print item.priority, item.index
                                isPredictable[item.index] = 0  # not predictable
                                if correction == True:
                                    correctionWeight = item.priority / sumForCorrection
                                    # correctedList = [x / correctionWeight for x in initial_X_test[item.index]]
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(correctionWeight)
                                else:
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(1.0)

                                unmodified_train_y.append(initial_y_test[item.index])
                                train_index_list.append(test_index_list[item.index])

                                loopDocList.append(int(initial_y_test[item.index]))
                                train_size_controller = train_size_controller + 1
                                # print X_train.append(X_test.pop(item.priority))

                        if protocol == 'SPL':
                            print "####SPL####"
                            randomArray = []
                            randomArrayIndex = 0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    randomArray.append(counter)
                                    randomArrayIndex = randomArrayIndex + 1
                            import random

                            random.shuffle(randomArray)

                            batch_counter = 0
                            for batch_counter in xrange(0, len(randomArray)):
                                #if batch_counter > len(randomArray) - 1:
                                #    break
                                if train_size_controller == size_limit:
                                    break
                                itemIndex = randomArray[batch_counter]
                                isPredictable[itemIndex] = 0
                                unmodified_train_X.append(initial_X_test[itemIndex])
                                unmodified_train_y.append(initial_y_test[itemIndex])
                                sampling_weight.append(1.0)

                                train_index_list.append(test_index_list[itemIndex])
                                loopDocList.append(int(initial_y_test[itemIndex]))
                                train_size_controller = train_size_controller + 1

                        if iter_sampling == True:
                            print "Oversampling in the active iteration list"
                            ros = RandomOverSampler()
                            initial_X_train = None
                            initial_y_train = None
                            initial_X_train, initial_y_train = ros.fit_sample(unmodified_train_X, unmodified_train_y)
                        else:
                            initial_X_train[:] = []
                            initial_y_train[:] = []
                            initial_X_train = copy.deepcopy(unmodified_train_X)
                            initial_y_train = copy.deepcopy(unmodified_train_y)

                        loopCounter = loopCounter + 1

                y_pred_all = {}

                human_label_str = ""

                for train_index in train_index_list:
                    y_pred_all[train_index] = y[train_index]
                    docNo = docIndex_DocNo[train_index]
                    human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(
                        y_pred_all[train_index]) + "\n"
                human_label_location_final = human_label_location + str(1.1) + '_human_.txt'
                text_file = open(human_label_location_final, "a")
                text_file.write(human_label_str)
                text_file.close()

                for train_index in xrange(0, len(X)):
                    if train_index not in train_index_list:
                        y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

                y_pred = []
                for key, value in y_pred_all.iteritems():
                    # print (key,value)
                    y_pred.append(value)

                f1score = f1_score(y, y_pred, average='binary')
                precision = precision_score(y, y_pred, average='binary')
                recall = recall_score(y, y_pred, average='binary')

                if train_per_centage_flag == True:
                    if (learning_curve.has_key(1.1)):
                        tmplist = learning_curve.get(1.1)
                        tmplist.append(f1score)
                        learning_curve[1.1] = tmplist
                    else:
                        tmplist = []
                        tmplist.append(f1score)
                        learning_curve[1.1] = tmplist

                    pred_topic_str = ""
                    for docIndex in xrange(0, len(X)):
                        docNo = docIndex_DocNo[docIndex]
                        pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(
                            y_pred_all[docIndex]) + "\n"

                    predicted_location_final = predicted_location_base + str(1.1) + '.txt'
                    text_file = open(predicted_location_final, "a")
                    text_file.write(pred_topic_str)
                    text_file.close()

                print "precision score:", precision
                print "recall score:", recall
                print "f-1 score:", f1score

                # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
                '''
                precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

                recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

                So now you are dividing 0/0.'''

                accuracy = accuracy_score(y, y_pred)
                print "model accuracy (%): ", accuracy * 100, "%"

                s = s + topic + "," + str(datasize) + "," + str(numberOne) + "," + str(numberZero) + "," + str(
                    prevelance) + "," + str(precision) + "," + str(recall) + "," + str(f1score) + "," + str(
                    accuracy) + "," + str(best_f1) + "\n";
                # writing the actual and prediction



                counter = 0
                # doing predcition again for all documents in the validation and test set for writing the prediction files
                # y_pred = model.predict(x_validation_and_test)
                for docIndex in xrange(0, len(X)):
                    docNo = docIndex_DocNo[docIndex]
                    pred_str = pred_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"
                    counter = counter + 1

                text_file = open(result_location, "w")
                text_file.write(s)
                text_file.close()

                text_file = open(predicted_location, "w")
                text_file.write(pred_str)
                text_file.close()

            else:
                initial_X_train = []
                initial_y_train = []
                train_index_list = []


                # collecting the seed list from Interactive Search (IS)
                seed_counter = 0
                seed_one_counter = 0
                seed_zero_counter = 0
                ask_for_label = 0


                for index in xrange(0,len(X)):
                    if y[index] == 1:
                        seed_one_counter = seed_one_counter + 1
                        train_index_list.append(index)
                        initial_X_train.append(X[index])
                        initial_y_train.append(y[index])
                        print seed_one_counter
                    if seed_one_counter == n_labeled/2 :
                        break

                for index in xrange(0,len(X)):

                    if y[index] == 0:
                        seed_zero_counter = seed_zero_counter + 1
                        train_index_list.append(index)
                        initial_X_train.append(X[index])
                        initial_y_train.append(y[index])
                        print seed_zero_counter
                    if seed_zero_counter == n_labeled/2:
                        break

                if seed_zero_counter != seed_one_counter:
                    skipList.append(topic)

                unmodified_train_X = copy.deepcopy(initial_X_train)
                unmodified_train_y = copy.deepcopy(initial_y_train)
                sampling_weight = []

                for sampling_index in xrange(0, len(initial_X_train)):
                    sampling_weight.append(1.0)


                if sampling == True: # sampling is for oversampling parameter
                    print "Oversampling in the seed list"
                    ros = RandomOverSampler()

                    initial_X_train_sampled, initial_y_train_sampled = ros.fit_sample(initial_X_train, initial_y_train)
                    initial_X_train = initial_X_train_sampled
                    initial_y_train = initial_y_train_sampled

                    initial_X_train = initial_X_train.tolist()
                    initial_y_train = initial_y_train.tolist()

                initial_X_test = []
                initial_y_test = []

                test_index_list = {}
                test_index_counter = 0
                for train_index in xrange(0, len(X)):
                    if train_index not in train_index_list:
                        initial_X_test.append(X[train_index])
                        test_index_list[test_index_counter] = train_index
                        test_index_counter = test_index_counter + 1
                        initial_y_test.append(y[train_index])

                print "Before Loop Lenght:", len(initial_X_train), len(initial_y_train)
                predictableSize = len(initial_X_test)
                isPredictable = [1] * predictableSize  # initially we will predict all

                loopCounter = 0
                best_model = 0
                learning_batch_size = n_labeled # starts with the seed size

                if train_per_centage_flag == False:
                    numberofloop = math.ceil(size / batch_size)
                    if numberofloop == 0:
                        numberofloop = 1
                    print "Number of loop", numberofloop

                    while loopCounter<=numberofloop:
                        print "Loop:", loopCounter

                        loopDocList = []

                        if protocol == 'SPL':
                            model = LogisticRegression(C=100000000)

                        print len(initial_X_train)
                        if correction == True:
                            model.fit(initial_X_train, initial_y_train, sample_weight=sampling_weight)
                        else:
                            model.fit(initial_X_train, initial_y_train)

                        y_pred_all = {}

                        for train_index in train_index_list:
                            y_pred_all[train_index] = y[train_index]

                        for train_index in xrange(0, len(X)):
                            if train_index not in train_index_list:
                                y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

                        y_pred = []
                        for key, value in y_pred_all.iteritems():
                            # print (key,value)
                            y_pred.append(value)

                        f1score = f1_score(y, y_pred, average='binary')

                        if (learning_curve.has_key(learning_batch_size)):
                            tmplist = learning_curve.get(learning_batch_size)
                            tmplist.append(f1score)
                            learning_curve[learning_batch_size] = tmplist
                        else:
                            tmplist = []
                            tmplist.append(f1score)
                            learning_curve[learning_batch_size] = tmplist

                        learning_batch_size = learning_batch_size + batch_size
                        precision = precision_score(y, y_pred, average='binary')
                        recall = recall_score(y, y_pred, average='binary')

                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        if isPredictable.count(1) == 0:
                            break

                        #if f1score == 1.0:
                        #    break
                        #    print "BREAKING LOOP BECAUSE F-1 is 1.0"

                        # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
                        '''
                        precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

                        recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

                        So now you are dividing 0/0.'''
                        # here is queueSize is the number of predictable element
                        queueSize = isPredictable.count(1)

                        if protocol == 'CAL':
                            print "####CAL####"
                            queue = Queue.PriorityQueue(queueSize)
                            y_prob = []
                            counter = 0
                            sumForCorrection = 0.0
                            for counter in xrange(0,predictableSize):
                                if isPredictable[counter] == 1:
                                    # reshapping reshape(1,-1) because it does not take one emelemt array
                                    # list does not contain reshape so we are using np,array
                                    # model.predit returns two value in index [0] of the list
                                    y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1,-1))[0]
                                    #y_prob = model.predict(initial_X_test[counter])
                                    #print y_prob
                                    queue.put(relevance(y_prob[1], counter))
                                    sumForCorrection = sumForCorrection + y_prob[1]


                            batch_counter = 0
                            while not queue.empty():
                                if batch_counter == batch_size:
                                    break
                                item = queue.get()
                                #print len(item)
                                #print item.priority, item.index

                                isPredictable[item.index] = 0 # not predictable
                                #initial_X_train.append(initial_X_test[item.index])
                                #initial_y_train.append(initial_y_test[item.index])

                                if correction == True:
                                    correctionWeight = item.priority / sumForCorrection
                                    #correctedItem = [x / correctionWeight for x in initial_X_test[item.index]]
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(correctionWeight)
                                else:
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(1.0)
                                unmodified_train_y.append(initial_y_test[item.index])

                                train_index_list.append(test_index_list[item.index])

                                #print "Docs:", initial_X_test[item.index]
                                loopDocList.append(int(initial_y_test[item.index]))
                                batch_counter = batch_counter + 1
                                #print X_train.append(X_test.pop(item.priority))

                        if protocol == 'SAL':
                            print "####SAL####"
                            queue = Queue.PriorityQueue(queueSize)
                            y_prob = []
                            counter = 0
                            sumForCorrection = 0.0
                            for counter in xrange(0,predictableSize):
                                if isPredictable[counter] == 1:
                                    # reshapping reshape(1,-1) because it does not take one emelemt array
                                    # list does not contain reshape so we are using np,array
                                    # model.predit returns two value in index [0] of the list
                                    y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1,-1))[0]
                                    #entropy = (-1)*(y_prob[0]*log(y_prob[0],2)+y_prob[1]*log(y_prob[1],2))
                                    entropy = calculate_entropy(y_prob[0], y_prob[1])
                                    queue.put(relevance(entropy, counter))
                                    sumForCorrection = sumForCorrection + entropy


                            batch_counter = 0
                            while not queue.empty():
                                if batch_counter == batch_size:
                                    break
                                item = queue.get()
                                #print len(item)
                                #print item.priority, item.index
                                isPredictable[item.index] = 0 # not predictable
                                if correction == True:
                                    correctionWeight = item.priority/sumForCorrection
                                    #correctedList = [x / correctionWeight for x in initial_X_test[item.index]]
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(correctionWeight)
                                else:
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(1.0)

                                unmodified_train_y.append(initial_y_test[item.index])
                                train_index_list.append(test_index_list[item.index])

                                loopDocList.append(int(initial_y_test[item.index]))
                                batch_counter = batch_counter + 1
                                #print X_train.append(X_test.pop(item.priority))



                        if protocol == 'SPL':
                            print "####SPL####"
                            randomArray =[]
                            randomArrayIndex = 0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    randomArray.append(counter)
                                    randomArrayIndex = randomArrayIndex + 1
                            import random
                            random.shuffle(randomArray)

                            batch_counter = 0
                            for batch_counter in xrange(0,batch_size):
                                if batch_counter>len(randomArray)-1:
                                    break
                                itemIndex = randomArray[batch_counter]
                                isPredictable[itemIndex] = 0
                                unmodified_train_X.append(initial_X_test[itemIndex])
                                unmodified_train_y.append(initial_y_test[itemIndex])
                                sampling_weight.append(1.0)

                                train_index_list.append(test_index_list[itemIndex])
                                loopDocList.append(int(initial_y_test[itemIndex]))


                        if iter_sampling == True:
                            print "Oversampling in the active iteration list"
                            ros = RandomOverSampler()
                            initial_X_train = None
                            initial_y_train = None
                            initial_X_train, initial_y_train = ros.fit_sample(unmodified_train_X, unmodified_train_y)
                        else:
                            initial_X_train[:] = []
                            initial_y_train[:] = []
                            initial_X_train = copy.deepcopy(unmodified_train_X)
                            initial_y_train = copy.deepcopy(unmodified_train_y)

                        loopCounter = loopCounter + 1
                else:
                    numberofloop = len(train_per_centage)
                    train_size_controller = len(initial_X_train)
                    while loopCounter < numberofloop:
                        size_limit = math.ceil(train_per_centage[loopCounter]*len(X))

                        print "Loop:", loopCounter
                        print "Initial size:",train_size_controller, "limit:", size_limit

                        loopDocList = []

                        if protocol == 'SPL':
                            model = LogisticRegression(C=100000000)

                        print len(initial_X_train)
                        if correction == True:
                            model.fit(initial_X_train, initial_y_train, sample_weight=sampling_weight)
                        else:
                            model.fit(initial_X_train, initial_y_train)

                        y_pred_all = {}

                        human_label_str = ""

                        for train_index in train_index_list:
                            y_pred_all[train_index] = y[train_index]
                            docNo = docIndex_DocNo[train_index]
                            human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[train_index]) + "\n"
                            #print y_pred_all[train_index]
                        human_label_location_final = human_label_location + str(train_per_centage[loopCounter]) + '_human_.txt'
                        text_file = open(human_label_location_final, "a")
                        text_file.write(human_label_str)
                        text_file.close()

                        for train_index in xrange(0, len(X)):
                            if train_index not in train_index_list:
                                y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

                        y_pred = []
                        for key, value in y_pred_all.iteritems():
                            # print (key,value)
                            y_pred.append(value)

                        pred_topic_str = ""
                        for docIndex in xrange(0, len(X)):
                            docNo = docIndex_DocNo[docIndex]
                            pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"
                        predicted_location_final = predicted_location_base + str(train_per_centage[loopCounter]) + '.txt'
                        text_file = open(predicted_location_final, "a")
                        text_file.write(pred_topic_str)
                        text_file.close()

                        f1score = f1_score(y, y_pred, average='binary')

                        if (learning_curve.has_key(train_per_centage[loopCounter])):
                            tmplist = learning_curve.get(train_per_centage[loopCounter])
                            tmplist.append(f1score)
                            learning_curve[train_per_centage[loopCounter]] = tmplist
                        else:
                            tmplist = []
                            tmplist.append(f1score)
                            learning_curve[train_per_centage[loopCounter]] = tmplist

                        learning_batch_size = learning_batch_size + batch_size
                        precision = precision_score(y, y_pred, average='binary')
                        recall = recall_score(y, y_pred, average='binary')

                        print "precision score:", precision
                        print "recall score:", recall
                        print "f-1 score:", f1score

                        if isPredictable.count(1) == 0:
                            break

                        # if f1score == 1.0:
                        #    break
                        #    print "BREAKING LOOP BECAUSE F-1 is 1.0"

                        # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
                        '''
                        precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

                        recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

                        So now you are dividing 0/0.'''
                        # here is queueSize is the number of predictable element
                        queueSize = isPredictable.count(1)

                        if protocol == 'CAL':
                            print "####CAL####"
                            queue = Queue.PriorityQueue(queueSize)
                            y_prob = []
                            counter = 0
                            sumForCorrection = 0.0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    # reshapping reshape(1,-1) because it does not take one emelemt array
                                    # list does not contain reshape so we are using np,array
                                    # model.predit returns two value in index [0] of the list
                                    y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))[0]
                                    # y_prob = model.predict(initial_X_test[counter])
                                    # print y_prob
                                    queue.put(relevance(y_prob[1], counter))
                                    sumForCorrection = sumForCorrection + y_prob[1]

                            batch_counter = 0
                            while not queue.empty():
                                if train_size_controller == size_limit:
                                    break
                                item = queue.get()
                                # print len(item)
                                # print item.priority, item.index

                                isPredictable[item.index] = 0  # not predictable
                                # initial_X_train.append(initial_X_test[item.index])
                                # initial_y_train.append(initial_y_test[item.index])

                                if correction == True:
                                    correctionWeight = item.priority / sumForCorrection
                                    # correctedItem = [x / correctionWeight for x in initial_X_test[item.index]]
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(correctionWeight)
                                else:
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(1.0)
                                unmodified_train_y.append(initial_y_test[item.index])

                                train_index_list.append(test_index_list[item.index])

                                # print "Docs:", initial_X_test[item.index]
                                loopDocList.append(int(initial_y_test[item.index]))
                                train_size_controller = train_size_controller + 1
                                # print X_train.append(X_test.pop(item.priority))

                        if protocol == 'SAL':
                            print "####SAL####"
                            queue = Queue.PriorityQueue(queueSize)
                            y_prob = []
                            counter = 0
                            sumForCorrection = 0.0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    # reshapping reshape(1,-1) because it does not take one emelemt array
                                    # list does not contain reshape so we are using np,array
                                    # model.predit returns two value in index [0] of the list
                                    y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))[0]
                                    #entropy = (-1) * (y_prob[0] * log(y_prob[0], 2) + y_prob[1] * log(y_prob[1], 2))
                                    entropy = calculate_entropy(y_prob[0], y_prob[1])
                                    queue.put(relevance(entropy, counter))
                                    sumForCorrection = sumForCorrection + entropy

                            batch_counter = 0
                            while not queue.empty():
                                if train_size_controller == size_limit:
                                    break
                                item = queue.get()
                                # print len(item)
                                # print item.priority, item.index
                                isPredictable[item.index] = 0  # not predictable
                                if correction == True:
                                    correctionWeight = item.priority / sumForCorrection
                                    # correctedList = [x / correctionWeight for x in initial_X_test[item.index]]
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(correctionWeight)
                                else:
                                    unmodified_train_X.append(initial_X_test[item.index])
                                    sampling_weight.append(1.0)

                                unmodified_train_y.append(initial_y_test[item.index])
                                train_index_list.append(test_index_list[item.index])

                                loopDocList.append(int(initial_y_test[item.index]))
                                train_size_controller = train_size_controller + 1
                                # print X_train.append(X_test.pop(item.priority))

                        if protocol == 'SPL':
                            print "####SPL####"
                            randomArray = []
                            randomArrayIndex = 0
                            for counter in xrange(0, predictableSize):
                                if isPredictable[counter] == 1:
                                    randomArray.append(counter)
                                    randomArrayIndex = randomArrayIndex + 1
                            import random

                            random.shuffle(randomArray)

                            batch_counter = 0
                            for batch_counter in xrange(0, len(randomArray)):
                                #if batch_counter > len(randomArray) - 1:
                                #    break
                                if train_size_controller == size_limit:
                                    break
                                itemIndex = randomArray[batch_counter]
                                isPredictable[itemIndex] = 0
                                unmodified_train_X.append(initial_X_test[itemIndex])
                                unmodified_train_y.append(initial_y_test[itemIndex])
                                sampling_weight.append(1.0)

                                train_index_list.append(test_index_list[itemIndex])
                                loopDocList.append(int(initial_y_test[itemIndex]))
                                train_size_controller = train_size_controller + 1

                        if iter_sampling == True:
                            print "Oversampling in the active iteration list"
                            ros = RandomOverSampler()
                            initial_X_train = None
                            initial_y_train = None
                            initial_X_train, initial_y_train = ros.fit_sample(unmodified_train_X, unmodified_train_y)
                        else:
                            initial_X_train[:] = []
                            initial_y_train[:] = []
                            initial_X_train = copy.deepcopy(unmodified_train_X)
                            initial_y_train = copy.deepcopy(unmodified_train_y)

                        loopCounter = loopCounter + 1


                print "Fininshed loop", len(initial_X_train)
                y_pred_all = {}

                human_label_str = ""

                for train_index in train_index_list:
                    y_pred_all[train_index] = y[train_index]
                    docNo = docIndex_DocNo[train_index]
                    human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(
                        y_pred_all[train_index]) + "\n"
                human_label_location_final = human_label_location + str(1.1) + '_human_.txt'
                text_file = open(human_label_location_final, "a")
                text_file.write(human_label_str)
                text_file.close()


                for train_index in xrange(0, len(X)):
                    if train_index not in train_index_list:
                        y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

                y_pred = []
                for key, value in y_pred_all.iteritems():
                    # print (key,value)
                    y_pred.append(value)

                f1score = f1_score(y, y_pred, average='binary')
                precision = precision_score(y, y_pred, average='binary')
                recall = recall_score(y, y_pred, average='binary')


                print "precision score:", precision
                print "recall score:", recall
                print "f-1 score:", f1score

                if train_per_centage_flag == True:
                    if (learning_curve.has_key(1.1)):
                        tmplist = learning_curve.get(1.1)
                        tmplist.append(f1score)
                        learning_curve[1.1] = tmplist
                    else:
                        tmplist = []
                        tmplist.append(f1score)
                        learning_curve[1.1] = tmplist

                    pred_topic_str = ""
                    for docIndex in xrange(0, len(X)):
                        docNo = docIndex_DocNo[docIndex]
                        pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(
                            y_pred_all[docIndex]) + "\n"

                    predicted_location_final = predicted_location_base + str(1.1) + '.txt'
                    text_file = open(predicted_location_final, "a")
                    text_file.write(pred_topic_str)
                    text_file.close()

                # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
                '''
                precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

                recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

                So now you are dividing 0/0.'''


                accuracy = accuracy_score(y, y_pred)
                print "model accuracy (%): ", accuracy * 100, "%"

                s = s + topic + "," + str(datasize) + "," + str(numberOne) + "," + str(numberZero) + "," + str(prevelance) + "," + str(precision) + "," + str(recall) + "," + str(f1score) + "," + str(accuracy) + "," + str(best_f1) +"\n";
                # writing the actual and prediction



                counter = 0
                # doing predcition again for all documents in the validation and test set for writing the prediction files
                #y_pred = model.predict(x_validation_and_test)
                for docIndex in xrange(0, len(X)):
                    docNo = docIndex_DocNo[docIndex]
                    pred_str = pred_str + str(topic) + " "+str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"
                    counter = counter + 1

            text_file = open(result_location, "w")
            text_file.write(s)
            text_file.close()

            #text_file = open(predicted_location, "w")
            #text_file.write(pred_str)
            #text_file.close()


for topic in skipList:
    print topic

s=""
for (key, valueList) in sorted(learning_curve.items()):
    size = len(valueList)
    sum = 0
    for value in valueList:
        sum = sum + value
    #print "value", sum/size
    s = s + str(sum/size) + ","

text_file = open(learning_curve_location, "w")
text_file.write(s)
text_file.close()

