'''Active learning for labeling the relevant document for TREC-8 dataset
@author: Md Mustafizur Rahman (nahid@utexas.edu)'''

import os
import numpy as np
import re
import math
import copy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#from sklearn.metrics import
from sklearn.linear_model import LogisticRegression
import collections
import time
import random

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
import pickle
from math import log
import Queue

import logging
logging.basicConfig()

TEXT_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/IndriData/'
RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/relevance.txt'
docrepresentation = "TF-IDF"  # can be BOW, TF-IDF
sampling=False # can be True or False
command_prompt_use = True

#if command_prompt_use == True:
import sys

datasource = sys.argv[1] # can be  dataset = ['TREC8', 'gov2', 'WT']
protocol = sys.argv[2]
lambda_param = float(sys.argv[3])
alpha_param = int(sys.argv[4]) # can be 1 or 2, 1 means more emphasize on easy topic, 2 means more emphasize on hard topic

'''
use_ranker = sys.argv[3]
iter_sampling = sys.argv[4]
correction = sys.argv[5] #'SAL' can be ['SAL', 'CAL', 'SPL']
train_per_centage_flag = sys.argv[6]
'''


# if deterministic is False then HT estimation have to True or Vice Versa
# Here deterministic = True means we sample a topic from a uniform distribution
deterministic = 'True' # this is hard coded and does not take as an input from command line
ht_estimation = 'False'


##############################
iter_sampling = 'False' # mean oversampling in active itearation phase
use_ranker = 'True' # Now we always use Ranker so keep it fixed at True
correction = 'False' # not used anymore but keep it fixed at Fasle
train_per_centage_flag = 'True' # not used anymore but keep it fixed at True

#lambda_param = 0.5
#alpha_param = 2 # can be 1 or 2, 1 means more emphasize on easy topic, 2 means more emphasize on hard topic

print "Ranker_use", use_ranker
print "iter_sampling", iter_sampling
print "correction", correction
print "train_percenetae", train_per_centage_flag

test_size = 0    # the percentage of samples in the dataset that will be
test_size_set = [0.2]
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#train_per_centage = [0.1, 0.2]

budget_list = []

ranker_location = {}
ranker_location["WT2013"] = "/work/04549/mustaf/maverick/data/TREC/WT2013/input.ICTNET13RSR2"
ranker_location["WT2014"] = "/work/04549/mustaf/maverick/data/TREC/WT2014/input.Protoss"
ranker_location["gov2"] = "/work/04549/mustaf/maverick/data/TREC/gov2/input.indri06AdmD"
ranker_location["TREC8"] = "/work/04549/mustaf/maverick/data/TREC/TREC8/input.ibmg99b"

n_labeled =  10 #50      # number of samples that are initially labeled
batch_size = 25 #50
preloaded = True
topicSkipList = [202,209,225, 237, 245, 255,269, 278, 803, 805] # remember to update the relevance file for this collection accordingly to TAU compute
#topicSkipList = [202,210,225,234,235,238,244,251,255,262,269,271,278,283,289,291,803,805]

skipList = []
topicBucketList = []
processed_file_location = ''
start_topic = 0
end_topic = 0
datasetsize = 0

import os

base_address = os.getcwd()+"/"
#base_address = "/work/04549/mustaf/maverick/data/TREC/"

if deterministic == 'True':
    deterministic = True
    #base_address = "/work/04549/mustaf/maverick/data/TREC/deterministic/"
    lambda_param = 0.0
    alpha_param = 1
else:
    deterministic = False

if ht_estimation == 'True':
    ht_estimation = True
    #base_address = "/work/04549/mustaf/maverick/data/TREC/estimation/"
else:
    ht_estimation = False

print "Base address:", base_address

if deterministic == True and ht_estimation == True:
    print("Both deterministic and htestimation parameter cannot be true in the same time")
    exit(-1)

base_address = base_address +str(datasource)+"/result/"
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
    base_address = base_address + "oversample/"
if correction == 'True':
    base_address = base_address + "htcorrection/"
    correction = True
if correction == 'False':
    correction = False

if train_per_centage_flag == 'True':
    train_per_centage_flag = True
else:
    train_per_centage_flag = False



if iter_sampling == True and correction == True:
    print "Over sampling and HT correction cannot be done together"
    exit(-1)

base_address = base_address + str(lambda_param) + "/" + str(alpha_param) + "/"

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
    datasetsize = 14474
else:
    processed_file_location = '/work/04549/mustaf/maverick/data/TREC/WT2014/processed_new.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/WT2014/qrelsadhoc2014.txt'
    start_topic = 251
    end_topic = 301
    datasetsize = 14432

all_reviews = {}
learning_curve = {} # per batch value for  validation set


class relevance(object):
    def __init__(self, priority, index):
        self.priority = priority
        self.index = index
        return
    def __cmp__(self, other):
        return -cmp(self.priority, other.priority)

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



if preloaded==False:
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        #print name
        #if name == "ft":
        path = os.path.join(TEXT_DATA_DIR, name)
        print path

        f = open(path)
        docNo = name[0:name.index('.')]
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
        for lines in f:
            if c < count:
                c = c + 1
                continue
            values = lines.split()
            c = c + 1
            tmpStr = tmpStr + " "+ str(values[2])
        print tmpStr
        #if docNo in docNo_label:
        all_reviews[docNo] = (review_to_words(tmpStr))
        f.close()
    output = open(processed_file_location, 'ab+')
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

# since we are skipping topics, we need to discard the documents and their document count also under that
# topic, which will be used to calculate the budget limit



# called initially get the distribution
def get_topic_distribution():
    total_document_in_relevance_judgement = 0
    topic_train_index_list = {} # key is the topic number and value is the train_index_list
    topic_initial_X_train = {}
    topic_initial_Y_train = {}
    topic_seed_one_counter = {}
    topic_estimated_one_counter = {}
    topic_estimated_zero_counter = {}
    topic_seed_zero_counter = {}
    topic_seed_counter = {}
    total_judged = 0
    topic_train_percentage_loop_counter = {}

    for topic in xrange(start_topic, end_topic):
        print "Topic:", topic
        if topic in topicSkipList:
            print "Skipping Topic :", topic
            continue
        topic = str(topic)

        topic_to_doclist = {}  # key is the topic(string) and value is the list of docNumber
        docNo_label = {}  # key is the DocNo and the value is the label
        docIndex_DocNo = {}  # key is the index used in my code value is the actual DocNo
        docNo_docIndex = {}  # key is the DocNo and the value is the index assigned by my code
        best_f1 = 0.0  # best f1 considering per iteraton of active learning
        print('Reading the relevance label')
        # file open
        f = open(RELEVANCE_DATA_DIR)
        print f
        tmplist = []
        for lines in f:
            values = lines.split()
            topicNo = values[0]
            if topicNo != topic:
                # print "Skipping", topic, topicNo
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
        # print len(topic_to_doclist)
        docList = topic_to_doclist[topic]
        print 'number of documents', len(docList)
        total_document_in_relevance_judgement = total_document_in_relevance_judgement + len(docList)
        # print docList
        # print ('Processing news text for topic number')
        relevance_label = []
        judged_review = []

        docIndex = 0
        for documentNo in docList:
            if all_reviews.has_key(documentNo):
                # print "in List", documentNo
                # print documentNo, 'len:', type(all_reviews[documentNo])

                # print all_reviews[documentNo]
                # exit(0)
                docIndex_DocNo[docIndex] = documentNo
                docNo_docIndex[documentNo] = docIndex
                docIndex = docIndex + 1
                judged_review.append(all_reviews[documentNo])
                relevance_label.append(docNo_label[documentNo])

        if docrepresentation == "TF-IDF":
            print "Using TF-IDF"
            vectorizer = TfidfVectorizer(analyzer="word", \
                                         tokenizer=None, \
                                         preprocessor=None, \
                                         stop_words=None, \
                                         max_features=15000)

            bag_of_word = vectorizer.fit_transform(judged_review)


        elif docrepresentation == "BOW":
            print "Uisng Bag of Word"
            vectorizer = CountVectorizer(analyzer="word", \
                                         tokenizer=None, \
                                         preprocessor=None, \
                                         stop_words=None, \
                                         max_features=15000)

            bag_of_word = vectorizer.fit_transform(judged_review)

        bag_of_word = bag_of_word.toarray()
        print bag_of_word.shape

        print "Bag of word completed"

        X = bag_of_word
        y = relevance_label

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

        print '----Started Training----'
        model = LogisticRegression()
        size = len(X) - n_labeled

        if size < 0:
            print "Train Size:", len(X), "seed:", n_labeled
            size = len(X)


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

        #seed_size_limit = math.ceil(train_per_centage[loopCounter] * len(X))
        #print "Initial Seed Limit", seed_size_limit
        seed_start = 0
        seed_counter = 0

        while seed_one_counter < 5:
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

        while seed_zero_counter < 5:
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

        topic_initial_X_train[topic] = initial_X_train
        topic_initial_Y_train[topic] = initial_y_train
        topic_train_index_list[topic] = train_index_list
        topic_seed_one_counter[topic] = seed_one_counter
        topic_seed_zero_counter[topic] = seed_zero_counter
        topic_seed_counter[topic] = seed_zero_counter + seed_one_counter
        total_judged = total_judged + topic_seed_counter[topic]
        topic_train_percentage_loop_counter[topic] = 0
        topic_estimated_one_counter[topic] = 0  # initiall no estimation
        topic_estimated_zero_counter[topic] = 0

    return topic_initial_X_train, topic_initial_Y_train, topic_seed_one_counter, topic_seed_zero_counter, topic_seed_counter, topic_train_index_list, total_judged, topic_train_percentage_loop_counter, total_document_in_relevance_judgement, topic_estimated_one_counter, topic_estimated_zero_counter



# need this mapper because we are skipping some topics in Gov2, WT13 and 14
indexToTopicId = {} # key is the index and value is the topicID
topicUsedListIndex = 0
for topic in xrange(start_topic, end_topic):
    if topic not in topicSkipList:
        indexToTopicId[topicUsedListIndex] = str(topic)
        topicUsedListIndex = topicUsedListIndex + 1

learning_curve_budget = {} # key is budget, valye is the average f-1 across topics
precision_curve_budget = {}
recall_curve_budget = {}
topic_complete_list = [] # list of topic with all document exhausted
topic_f1_scores = {}
topic_precision_scores = {}
topic_recall_scores = {}
topic_pred_str = {}
topic_human_str = {}
#topic_complete_list = len(end_topic - start_topic + 1)
topic_complete_list = [0] * (topicUsedListIndex)  # initially we will use all topic, when a index == 1 we skip that topic
budget_limit_to_train_percentage_mapper = {}

for test_size in test_size_set:
    seed = 1335
    for fold in xrange(1,2):
        #np.random.seed(seed)
        np.random.seed(seed=int(time.time()))
	seed = seed + fold
        result_location = base_address + 'result_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location_base = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_'
        human_label_location = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_'

        learning_curve_location = base_address + 'learning_curve_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        recall_curve_location = base_address + 'recall_curve_protocol:' + protocol + '_batch:' + str(
            batch_size) + '_seed:' + str(n_labeled) + '_fold' + str(fold) + '.txt'
        precision_curve_location = base_address + 'precision_curve_protocol:' + protocol + '_batch:' + str(
            batch_size) + '_seed:' + str(n_labeled) + '_fold' + str(fold) + '.txt'

        s = "";
        pred_str = ""


        topic_initial_X_train, topic_initial_Y_train, topic_seed_one_counter, topic_seed_zero_counter, topic_seed_counter, topic_train_index_list, total_judged, topic_loopCounter, total_document_in_relevance_judgement, topic_estimated_one_counter, topic_estimated_zero_counter  = get_topic_distribution()

        print "Collection:", datasource, "Actual Files used:", total_document_in_relevance_judgement
        # calculating the budget size
        for percentage in train_per_centage:
            budget_list.append(math.floor(percentage * total_document_in_relevance_judgement))
            budget_limit_to_train_percentage_mapper[math.floor(percentage * total_document_in_relevance_judgement)] = percentage
        print "Budget list"

        # dummy budget limit so that all the data get written
        budget_list.append(total_document_in_relevance_judgement+1000)
        budget_limit_to_train_percentage_mapper[total_document_in_relevance_judgement+1000] = 1.01

        for budget_limit in budget_list:
            print budget_limit


        topic_distribution = []
        for topic in xrange(start_topic, end_topic):
            print "Topic:", topic
            if topic in topicSkipList:
                print "Skipping Topic :", topic
                continue
            topic = str(topic)
            print topic_seed_one_counter[topic], topic_seed_zero_counter[topic], topic_seed_counter[topic]
            alpha =  0.0
            if alpha_param == 1:
                alpha = ((topic_seed_one_counter[topic] * 1.0) / topic_seed_counter[topic])
            else: # alpha_param == 2
                alpha = 1.0 - ((topic_seed_one_counter[topic] * 1.0) / topic_seed_counter[topic])
            beta = 1- ((topic_seed_counter[topic]*1.0)/total_judged)
            topic_selection_probability = alpha*lambda_param + beta * (1 - lambda_param)
            topic_distribution.append(topic_selection_probability)




        print "topic distribution", topic_distribution
        normalized_topic_distribution = [float(topic_prob) / sum(topic_distribution) for topic_prob in topic_distribution]
        print normalized_topic_distribution
        print "sanity check, ", sum(normalized_topic_distribution)

        number_of_samples = 0
        # used for deterministic topic selection
        deterministic_topic_sample_number = 0
        for budget_limit in budget_list:
            if sum(topic_distribution) == 0.0:
                print "ALL Topics are exhausted but budget not finished"
                break
            while total_judged < budget_limit:

                # sampling from topic distribution
                print "Current Budget Limit:", budget_limit, "total judged", total_judged

                sample_topic_list = -1
                if deterministic == False:
                    sample_topic_list = np.random.multinomial(1, normalized_topic_distribution, size=1)[0].tolist()
                    topic = indexToTopicId[sample_topic_list.index(1)]
                else:
                    print "Round Robin Topic Selection"
                    # Deterministic
                    sample_topic_list = deterministic_topic_sample_number%topicUsedListIndex
                    topic = indexToTopicId[sample_topic_list]
                    deterministic_topic_sample_number = deterministic_topic_sample_number + 1

                    #print "Sampling topic from a random uniform distribution"
                    # uniform distribution
                    #uniformDistribution = [1.0 / topicUsedListIndex] * topicUsedListIndex
                    #print uniformDistribution

                    #sample_topic_list = np.random.multinomial(1, uniformDistribution, size=1)[0].tolist()
                    #topic = indexToTopicId[sample_topic_list.index(1)]

                print "Sampled Topic:", topic
                #number_of_samples = number_of_samples + 1
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
                for lines in f:
                    values = lines.split()
                    #print lines

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
                print len(topic_to_doclist)
                docList = topic_to_doclist[str(topic)]
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
                # print "Number of zero", numberZero


                '''
                print type(X)
                print X[0]
                print len(X[0])
                print type(y)
                X = pd.DataFrame(bag_of_word)
                y = pd.Series(relevance_label)

                print type(X)
                print len(X)
                '''
                #exit(0)
                print "=========Before Sampling======"

                print "Whole Dataset size: ", datasize
                print "Number of Relevant", numberOne
                print "Number of non-relevant", numberZero
                print "prevelance ratio", prevelance * 100

                #print "After", y_train


                print '----Started Training----'
                model = LogisticRegression()
                size = len(X) - n_labeled

                if size<0:
                    print "Train Size:", len(X) , "seed:", n_labeled
                    size = len(X)

                initial_X_train = topic_initial_X_train[topic]
                initial_y_train = topic_initial_Y_train[topic]

                train_index_list = topic_train_index_list[topic]
                print "Topic:", topic, "train_index_list", len(train_index_list)
                loopCounter = topic_loopCounter[topic]
                currentTopic = topic
                topic_total_document_judged = topic_seed_counter[topic] # total document judged so far for the topic

                if topic_total_document_judged<len(X): # topic still has document
                    if (topic_total_document_judged + 25) < len(X):
                        batch_size = 25
                    else:
                        batch_size = len(X) - topic_total_document_judged

                    if use_ranker == True:
                        # collecting the seed list from the Rankers
                        seed_list = Ranker_topic_to_doclist[topic]
                        seed_counter = 0
                        seed_one_counter = 0
                        seed_zero_counter = 0
                        ask_for_label = 0

                        #seed_size_limit = math.ceil(train_per_centage[loopCounter] * len(X))
                        #print "Initial Seed Limit", seed_size_limit
                        seed_start = 0
                        seed_counter = 0

                        unmodified_train_X = copy.deepcopy(initial_X_train)
                        unmodified_train_y = copy.deepcopy(initial_y_train)
                        sampling_weight = []

                        for sampling_index in xrange(0, len(initial_X_train)):
                            sampling_weight.append(1.0)

                        # Ranker needs oversampling, but when HTCorrection true we cannot perform oversample
                        if use_ranker == True and iter_sampling == True:
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


                        print "Before Loop Lenght:", len(initial_X_train), len(initial_y_train)
                        predictableSize = len(initial_X_test)
                        isPredictable = [1] * predictableSize  # initially we will predict all

                        #loopCounter = 0
                        best_model = 0
                        learning_batch_size = n_labeled  # starts with the seed size

                        if train_per_centage_flag == False:
                            numberofloop = math.ceil(size / batch_size)
                            if numberofloop == 0:
                                numberofloop = 1
                            print "Number of loop", numberofloop

                            while loopCounter <= numberofloop:
                                print "Loop:", loopCounter

                                loopDocList = []

                                if protocol == 'SPL':
                                    model = LogisticRegression()

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
                                            entropy = (-1) * (y_prob[0] * log(y_prob[0], 2) + y_prob[1] * log(y_prob[1], 2))
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
                            estimated_remaining_relevant_documnets = 0
                            estimated_remaining_non_relevant_documnets = 0

                            #size_limit = math.ceil(train_per_centage[loopCounter]*len(X))
                            size_limit = train_size_controller + batch_size
                            print "Loop:", loopCounter
                            print "Initial size:",train_size_controller, "limit:", size_limit

                            loopDocList = []

                            if protocol == 'SPL':
                                model = LogisticRegression()

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
                            '''
                            human_label_location_final = human_label_location + str(budget_limit) + '_human_.txt'
                            text_file = open(human_label_location_final, "a")
                            text_file.write(human_label_str)
                            text_file.close()
                            '''
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

                            '''
                            predicted_location_final = predicted_location_base + str(budget_limit) + '.txt'
                            text_file = open(predicted_location_final, "a")
                            text_file.write(pred_topic_str)
                            text_file.close()

                            '''

                            f1score = f1_score(y, y_pred, average='binary')
                            precision = precision_score(y, y_pred, average='binary')
                            recall = recall_score(y, y_pred, average='binary')

                            topic_f1_scores[topic] = f1score
                            topic_precision_scores[topic] = precision
                            topic_recall_scores[topic] = recall

                            topic_human_str[topic] = human_label_str
                            topic_pred_str[topic] = pred_topic_str

                            if (learning_curve.has_key(budget_limit_to_train_percentage_mapper[budget_limit])):
                                tmplist = learning_curve.get(budget_limit_to_train_percentage_mapper[budget_limit])
                                tmplist.append(f1score)
                                learning_curve[budget_limit_to_train_percentage_mapper[budget_limit]] = tmplist
                            else:
                                tmplist = []
                                tmplist.append(f1score)
                                learning_curve[budget_limit_to_train_percentage_mapper[budget_limit]] = tmplist

                            learning_batch_size = learning_batch_size + batch_size


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

                            elementsProbability = []
                            elementsIndex = []
                            elementsLabel = []

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
                                        elementsProbability.append(y_prob[1])
                                        elementsIndex.append(counter)
                                        elementsLabel.append(initial_y_test[counter])

                                if ht_estimation == True or deterministic == True:
                                    print "Deterministic:", deterministic, "Inside CAL"
                                    normalized_element_probability = [float(elem_prob) / sum(elementsProbability)
                                                                      for
                                                                      elem_prob in
                                                                      elementsProbability]

                                    # print "normalized element prob:", normalized_element_probability
                                    sample_document_list = \
                                    np.random.multinomial(batch_size, normalized_element_probability, size=1)[
                                        0].tolist()

                                    # print "sample document list:", sample_document_list
                                    # sample_document_list = [1,0,3] # mean oth index 1 times, second index 3 times
                                    # so next line check documentValue >= 1

                                    document_id_list = []
                                    document_id_list = [documentid for documentid, documentvalue in
                                                        enumerate(sample_document_list) if documentvalue >= 1]
                                    # calculate inclusion probability only for relevant documents
                                    # to save compuatation tme
                                    # print "document id list:", document_id_list
                                    # unique document list
                                    unique_document_id_list = []
                                    unique_document_id_list = list(set(document_id_list))
                                    # print "unique docs list:", unique_document_id_list
                                    # document_inclusion_probability = []


                                    for documentid in unique_document_id_list:
                                        # print elementsLabel
                                        if int(elementsLabel[documentid]) == 1:
                                            inclusion_prob = 1 - pow((1 - normalized_element_probability[documentid]), batch_size)
                                            estimated_remaining_relevant_documnets = estimated_remaining_relevant_documnets + (elementsLabel[documentid]/inclusion_prob)

                                        else:
                                            inclusion_prob = 1 - pow((1 - normalized_element_probability[documentid]), batch_size)
                                            estimated_remaining_non_relevant_documnets = estimated_remaining_non_relevant_documnets + (elementsLabel[documentid]/inclusion_prob)

                                        itemIndex = elementsIndex[documentid]

                                        isPredictable[itemIndex] = 0  # not predictable
                                        # initial_X_train.append(initial_X_test[item.index])
                                        # initial_y_train.append(initial_y_test[item.index])

                                        unmodified_train_X.append(initial_X_test[itemIndex])
                                        unmodified_train_y.append(initial_y_test[itemIndex])

                                        if int(initial_y_test[itemIndex]) == 1:
                                            seed_one_counter = seed_one_counter + 1
                                        else:
                                            seed_zero_counter = seed_zero_counter + 1

                                        train_index_list.append(test_index_list[itemIndex])

                                        # print "Docs:", initial_X_test[item.index]
                                        loopDocList.append(int(initial_y_test[itemIndex]))
                                        train_size_controller = train_size_controller + 1
                                        # print X_train.append(X_test.pop(item.priority))

                                    print "Estimated Relevant Documents:", estimated_remaining_relevant_documnets
                                    # updating batc_size since we might not use 25 since we are performing sample with replacement
                                    batch_size = len(unique_document_id_list)
                                else:

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
                                        if int(initial_y_test[item.index]) == 1:
                                            seed_one_counter = seed_one_counter + 1
                                        else:
                                            seed_zero_counter = seed_zero_counter + 1

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
                                        entropy = (-1) * (y_prob[0] * log(y_prob[0], 2) + y_prob[1] * log(y_prob[1], 2))
                                        queue.put(relevance(entropy, counter))
                                        sumForCorrection = sumForCorrection + entropy
                                        elementsProbability.append(entropy)
                                        elementsIndex.append(counter)
                                        elementsLabel.append(initial_y_test[counter])

                                if ht_estimation == True or deterministic == True:
                                    print "Deterministic:", deterministic, "Inside SAL"
                                    normalized_element_probability = [float(elem_prob) / sum(elementsProbability)
                                                                      for
                                                                      elem_prob in
                                                                      elementsProbability]

                                    # print "normalized element prob:", normalized_element_probability
                                    sample_document_list = \
                                        np.random.multinomial(batch_size, normalized_element_probability, size=1)[
                                            0].tolist()

                                    # print "sample document list:", sample_document_list
                                    # sample_document_list = [1,0,3] # mean oth index 1 times, second index 3 times
                                    # so next line check documentValue >= 1

                                    document_id_list = []
                                    document_id_list = [documentid for documentid, documentvalue in
                                                        enumerate(sample_document_list) if documentvalue >= 1]
                                    # calculate inclusion probability only for relevant documents
                                    # to save compuatation tme
                                    # print "document id list:", document_id_list
                                    # unique document list
                                    unique_document_id_list = []
                                    unique_document_id_list = list(set(document_id_list))
                                    # print "unique docs list:", unique_document_id_list
                                    # document_inclusion_probability = []


                                    for documentid in unique_document_id_list:
                                        # print elementsLabel
                                        if int(elementsLabel[documentid]) == 1:
                                            inclusion_prob = 1 - pow((1 - normalized_element_probability[documentid]), batch_size)
                                            estimated_remaining_relevant_documnets = estimated_remaining_relevant_documnets + (elementsLabel[documentid]/inclusion_prob)

                                        else:
                                            inclusion_prob = 1 - pow((1 - normalized_element_probability[documentid]), batch_size)
                                            estimated_remaining_non_relevant_documnets = estimated_remaining_non_relevant_documnets + (elementsLabel[documentid]/inclusion_prob)

                                        itemIndex = elementsIndex[documentid]

                                        isPredictable[itemIndex] = 0  # not predictable
                                        # initial_X_train.append(initial_X_test[item.index])
                                        # initial_y_train.append(initial_y_test[item.index])

                                        unmodified_train_X.append(initial_X_test[itemIndex])
                                        unmodified_train_y.append(initial_y_test[itemIndex])

                                        if int(initial_y_test[itemIndex]) == 1:
                                            seed_one_counter = seed_one_counter + 1
                                        else:
                                            seed_zero_counter = seed_zero_counter + 1

                                        train_index_list.append(test_index_list[itemIndex])

                                        # print "Docs:", initial_X_test[item.index]
                                        loopDocList.append(int(initial_y_test[itemIndex]))
                                        train_size_controller = train_size_controller + 1
                                        # print X_train.append(X_test.pop(item.priority))

                                    print "Estimated Relevant Documents:", estimated_remaining_relevant_documnets
                                    # updating batc_size since we might not use 25 since we are performing sample with replacement
                                    batch_size = len(unique_document_id_list)
                                else:

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
                                        if int(initial_y_test[item.index]) == 1:
                                            seed_one_counter = seed_one_counter + 1
                                        else:
                                            seed_zero_counter = seed_zero_counter + 1

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
                                    if int(initial_y_test[itemIndex]) == 1:
                                        seed_one_counter = seed_one_counter + 1
                                    else:
                                        seed_zero_counter = seed_zero_counter + 1

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

                            topic_initial_X_train[topic] = unmodified_train_X
                            topic_initial_Y_train[topic] = unmodified_train_y
                            topic_seed_one_counter[topic] =  topic_seed_one_counter[topic] + seed_one_counter

                            topic_estimated_one_counter[topic] = estimated_remaining_relevant_documnets  # it is only update no increment
                            topic_estimated_zero_counter[topic] = estimated_remaining_non_relevant_documnets

                            topic_seed_zero_counter[topic] =  topic_seed_zero_counter[topic] + seed_zero_counter
                            topic_seed_counter[topic] = topic_seed_counter[topic] + seed_one_counter + seed_zero_counter
                            topic_train_index_list[topic] = train_index_list
                            topic_loopCounter[topic] = topic_loopCounter[topic] + 1
                            print "Topic Seed Counter: It should be 25", batch_size, topic_seed_counter[topic]
                            total_judged = total_judged + batch_size
                    topic_distribution = []
                    for topic in xrange(start_topic, end_topic):
                        print "Topic:", topic
                        if topic in topicSkipList:
                            print "Skipping Topic :", topic
                            continue
                        topic = str(topic)
                        print topic_seed_one_counter[topic], topic_seed_zero_counter[topic], topic_seed_counter[topic]

                        alpha = 0.0

                        if ht_estimation == False:
                            if alpha_param == 1:
                                alpha = ((topic_seed_one_counter[topic] * 1.0) / topic_seed_counter[topic])
                            else:  # alpha_param == 2
                                alpha = 1.0 - ((topic_seed_one_counter[topic] * 1.0) / topic_seed_counter[topic])
                        else:
                            numerator = topic_seed_one_counter[topic] + topic_estimated_one_counter[topic]
                            denumerator = topic_seed_one_counter[topic] + topic_estimated_one_counter[topic] + \
                                          topic_seed_zero_counter[topic] + topic_estimated_zero_counter[topic]

                            if alpha_param == 1:
                                alpha = (numerator * 1.0) / denumerator
                            else:  # alpha_param == 2
                                alpha = 1.0 - ((numerator * 1.0) / denumerator)

                        beta = 1- ((topic_seed_counter[topic] * 1.0) / total_judged)
                        topic_selection_probability = alpha * lambda_param + beta * (1 - lambda_param)
                        topic_distribution.append(topic_selection_probability)

                    # if any topic is complete make it's probability 0.0
                    indices = [index for index, topicIsComplete in enumerate(topic_complete_list) if topicIsComplete == 1]
                    for index_complete in indices:
                        topic_distribution[index_complete] = 0.0

                    #if sum(topic_distribution) == 0.0:
                    #    break

                    print "topic distribution", topic_distribution
                    normalized_topic_distribution = [float(topic_prob) / sum(topic_distribution) for topic_prob in
                                                     topic_distribution]
                    print normalized_topic_distribution
                    print "sanity check, ", sum(normalized_topic_distribution)

                else:
                    print "topic:", topic, "get Finished"


                    #################

                    y_pred_all = {}

                    human_label_str = ""

                    for train_index in train_index_list:
                        y_pred_all[train_index] = y[train_index]
                        docNo = docIndex_DocNo[train_index]
                        human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(
                            y_pred_all[train_index]) + "\n"
                    '''
                    human_label_location_final = human_label_location + str(budget_limit) + '_human_.txt'
                    text_file = open(human_label_location_final, "a")
                    text_file.write(human_label_str)
                    text_file.close()
                    '''
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
                        pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(
                            y_pred_all[docIndex]) + "\n"

                    '''
                    predicted_location_final = predicted_location_base + str(budget_limit) + '.txt'
                    text_file = open(predicted_location_final, "a")
                    text_file.write(pred_topic_str)
                    text_file.close()

                    '''

                    f1score = f1_score(y, y_pred, average='binary')
                    precision = precision_score(y, y_pred, average='binary')
                    recall = recall_score(y, y_pred, average='binary')

                    topic_f1_scores[topic] = f1score
                    topic_precision_scores[topic] = precision
                    topic_recall_scores[topic] = recall

                    topic_human_str[topic] = human_label_str
                    topic_pred_str[topic] = pred_topic_str

                    #################



                    if deterministic == True:
                        topicIndexNumber = sample_topic_list
                    else:
                        topicIndexNumber = sample_topic_list.index(1) # this is topic index (0,1,...,48) not topic number (401, ...450)

                    topic_complete_list[topicIndexNumber] = 1 # topic is complete ,so mark it as 1

                    topic_distribution = []
                    for topic in xrange(start_topic, end_topic):
                        print "Topic:", topic
                        if topic in topicSkipList:
                            print "Skipping Topic :", topic
                            continue
                        topic = str(topic)
                        print topic_seed_one_counter[topic], topic_seed_zero_counter[topic], topic_seed_counter[topic]
                        alpha = 0.0

                        if ht_estimation == False:
                            if alpha_param == 1:
                                alpha = ((topic_seed_one_counter[topic] * 1.0) / topic_seed_counter[topic])
                            else:  # alpha_param == 2
                                alpha = 1.0 - ((topic_seed_one_counter[topic] * 1.0) / topic_seed_counter[topic])
                        else:
                            numerator = topic_seed_one_counter[topic] + topic_estimated_one_counter[topic]
                            denumerator = topic_seed_one_counter[topic] + topic_estimated_one_counter[topic] + \
                                          topic_seed_zero_counter[topic] + topic_estimated_zero_counter[topic]

                            if alpha_param == 1:
                                alpha = (numerator * 1.0) / denumerator
                            else:  # alpha_param == 2
                                alpha = 1.0 - ((numerator * 1.0) / denumerator)

                        beta = 1- ((topic_seed_counter[topic] * 1.0) / total_judged)
                        topic_selection_probability = alpha * lambda_param + beta * (1 - lambda_param)
                        if topic == currentTopic:
                            topic_selection_probability = 0.0
                        topic_distribution.append(topic_selection_probability)

                    # if any topic is complete make it's probability 0.0
                    indices = [index for index, topicIsComplete in enumerate(topic_complete_list) if
                               topicIsComplete == 1]
                    for index_complete in indices:
                        topic_distribution[index_complete] = 0.0

                    if sum(topic_distribution) == 0.0:
                        break

                    print "topic distribution", topic_distribution
                    normalized_topic_distribution = [float(topic_prob) / sum(topic_distribution) for topic_prob in
                                                     topic_distribution]
                    print normalized_topic_distribution
                    print "sanity check, ", sum(normalized_topic_distribution)

            print "Budget Limit:", budget_limit, "Total Judged:", total_judged, "Reached need to STORE DATA"

            sum_f1 = 0.0
            sum_precision = 0.0
            sum_recall = 0.0
            human_label = ""
            pred_label = ""
            number_of_topic = 0
            topic_budget_allocation = ""
            for topic in xrange(start_topic, end_topic):
                if topic in topicSkipList:
                    continue
                topic = str(topic)
                if topic_f1_scores.has_key(topic):
                    sum_f1 = sum_f1 + topic_f1_scores[topic]
                    sum_precision = sum_precision + topic_precision_scores[topic]
                    sum_recall = sum_recall + topic_recall_scores[topic]
                if topic_human_str.has_key(topic):
                    human_label = human_label + topic_human_str[topic]
                if topic_pred_str.has_key(topic):
                    pred_label = pred_label +  topic_pred_str[topic]
                number_of_topic = number_of_topic + 1
                topic_budget_allocation = topic_budget_allocation + str(topic_seed_counter[topic]) + ","


            average_f1_across_topic = (sum_f1*1.0)/number_of_topic
            learning_curve_budget[budget_limit_to_train_percentage_mapper[budget_limit]] = average_f1_across_topic

            average_precision_across_topic = (sum_precision*1.0)/number_of_topic
            precision_curve_budget[budget_limit_to_train_percentage_mapper[budget_limit]] = average_precision_across_topic

            average_recall_across_topic = (sum_recall * 1.0) / number_of_topic
            recall_curve_budget[budget_limit_to_train_percentage_mapper[budget_limit]] = average_recall_across_topic

            human_label_location_final = human_label_location + str(budget_limit_to_train_percentage_mapper[budget_limit]) + '_human_.txt'
            text_file = open(human_label_location_final, "w")
            text_file.write(human_label)
            text_file.close()

            predicted_location_final = predicted_location_base + str(budget_limit_to_train_percentage_mapper[budget_limit]) + '.txt'
            text_file = open(predicted_location_final, "w")
            text_file.write(pred_label)
            text_file.close()

            budget_location_final = predicted_location_base + str(budget_limit_to_train_percentage_mapper[budget_limit]) + '_budget.txt'
            text_file = open(budget_location_final, "w")
            text_file.write(topic_budget_allocation)
            text_file.close()


# predicting and saving all the last step values

print "Last PRINT"
topic = int(topic)
for topic in xrange(start_topic, end_topic):
    print "INSIDE Last loop"
    if topic in topicSkipList:
        print "Skipping Topic :", topic
        continue
    topic = str(topic)

    topic_to_doclist = {}  # key is the topic(string) and value is the list of docNumber
    docNo_label = {}  # key is the DocNo and the value is the label
    docIndex_DocNo = {}  # key is the index used in my code value is the actual DocNo
    docNo_docIndex = {}  # key is the DocNo and the value is the index assigned by my code
    best_f1 = 0.0  # best f1 considering per iteraton of active learning
    print('Reading the relevance label')
    # file open
    f = open(RELEVANCE_DATA_DIR)
    print f
    tmplist = []
    for lines in f:
        values = lines.split()
        # print lines

        topicNo = values[0]

        if topicNo != topic:
            # print "Skipping", topic, topicNo
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
    print len(topic_to_doclist)
    docList = topic_to_doclist[str(topic)]
    print 'number of documents', len(docList)
    # print docList
    # print ('Processing news text for topic number')
    relevance_label = []
    judged_review = []

    docIndex = 0
    for documentNo in docList:
        if all_reviews.has_key(documentNo):
            # print "in List", documentNo
            # print documentNo, 'len:', type(all_reviews[documentNo])

            # print all_reviews[documentNo]
            # exit(0)
            docIndex_DocNo[docIndex] = documentNo
            docNo_docIndex[documentNo] = docIndex
            docIndex = docIndex + 1
            judged_review.append(all_reviews[documentNo])
            relevance_label.append(docNo_label[documentNo])

    if docrepresentation == "TF-IDF":
        print "Using TF-IDF"
        vectorizer = TfidfVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=None, \
                                     max_features=15000)

        bag_of_word = vectorizer.fit_transform(judged_review)


    elif docrepresentation == "BOW":
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        print "Uisng Bag of Word"
        vectorizer = CountVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=None, \
                                     max_features=15000)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        bag_of_word = vectorizer.fit_transform(judged_review)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    bag_of_word = bag_of_word.toarray()
    print bag_of_word.shape
    # vocab = vectorizer.get_feature_names()
    # print vocab

    # Sum up the counts of each vocabulary word
    # dist = np.sum(bag_of_word, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    # for tag, count in zip(vocab, dist):
    #    print count, tag

    print "Bag of word completed"

    X = bag_of_word
    y = relevance_label

    initial_X_train = topic_initial_X_train[topic]
    initial_y_train = topic_initial_Y_train[topic]
    train_index_list = topic_train_index_list[topic]
    print "Topic:", topic, "train_index_list", len(train_index_list)

    model = LogisticRegression()
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

    for train_index in xrange(0, len(X)):
        if train_index not in train_index_list:
            y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

    y_pred = []
    for key, value in y_pred_all.iteritems():
        y_pred.append(value)

    pred_topic_str = ""
    for docIndex in xrange(0, len(X)):
        docNo = docIndex_DocNo[docIndex]
        pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"

    f1score = f1_score(y, y_pred, average='binary')

    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')

    topic_f1_scores[topic] = f1score
    topic_precision_scores[topic] = precision
    topic_recall_scores[topic] = recall

    topic_human_str[topic] = human_label_str
    topic_pred_str[topic] = pred_topic_str


    print "precision score:", precision
    print "recall score:", recall
    print "f-1 score:", f1score



sum_f1 = 0.0
sum_precision = 0.0
sum_recall = 0.0
human_label = ""
pred_label = ""
number_of_topic = 0
topic_budget_allocation = ""
topic = int(topic)
for topic in xrange(start_topic, end_topic):
    if topic in topicSkipList:
        continue
    topic = str(topic)
    if topic_f1_scores.has_key(topic):
        sum_f1 = sum_f1 + topic_f1_scores[topic]
        sum_precision = sum_precision + topic_precision_scores[topic]
        sum_recall = sum_recall + topic_recall_scores[topic]
    if topic_human_str.has_key(topic):
        human_label = human_label + topic_human_str[topic]
    if topic_pred_str.has_key(topic):
        pred_label = pred_label +  topic_pred_str[topic]
    number_of_topic = number_of_topic + 1
    topic_budget_allocation = topic_budget_allocation + str(topic_seed_counter[topic]) + ","


average_f1_across_topic = (sum_f1*1.0)/number_of_topic
learning_curve_budget[1.1] = average_f1_across_topic

average_precision_across_topic = (sum_precision*1.0)/number_of_topic
precision_curve_budget[1.1] = average_precision_across_topic

average_recall_across_topic = (sum_recall * 1.0) / number_of_topic
recall_curve_budget[1.1] = average_recall_across_topic

human_label_location_final = human_label_location + str(1.1) + '_human_.txt'
text_file = open(human_label_location_final, "w")
text_file.write(human_label)
text_file.close()

predicted_location_final = predicted_location_base + str(1.1) + '.txt'
text_file = open(predicted_location_final, "w")
text_file.write(pred_label)
text_file.close()

budget_location_final = predicted_location_base + str(1.1) + '_budget.txt'
text_file = open(budget_location_final, "w")
text_file.write(topic_budget_allocation)
text_file.close()

print "TOPIC Complete List"
for topic in topic_complete_list:
    print topic


s=""
for (key, value) in sorted(learning_curve_budget.items()):
    s = s + str(value) + ","
text_file = open(learning_curve_location, "w")
text_file.write(s)
text_file.close()



s=""
for (key, value) in sorted(precision_curve_budget.items()):
    s = s + str(value) + ","
text_file = open(precision_curve_location, "w")
text_file.write(s)
text_file.close()


s=""
for (key, value) in sorted(recall_curve_budget.items()):
    s = s + str(value) + ","
text_file = open(recall_curve_location, "w")
text_file.write(s)
text_file.close()
