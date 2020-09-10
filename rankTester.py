from tqdm import tqdm
from gensim import corpora, similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.similarities import Similarity

from os import listdir
from os.path import isfile, join

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

#from imblearn.ensemble import BalancedBaggingClassifier
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import random
import copy
import argparse
import scipy.sparse as sp
import numpy as np
import time
import math
from random import randint
import Queue
import pickle
import os
from multiprocessing import Pool as ProcessPool
import itertools
from functools import partial



# import user pythons file
from topic_description import TRECTopics
from systemReader import systemReader
from global_definition import *
from qRelsProcessor import *

rng = np.random.seed(0)




# multiprocess function should be GLOBAL it cannot be under a class!
def construct_predicted_qrels_multi_processsing(topicId, classifier, classifier_name, document_collection,
                                                docIdToDocIndex, topic_qrels):
    # document_to_label is a dictionary of document related to topicId
    document_to_label = topic_qrels[topicId]
    total_documents = len(docIdToDocIndex)
    train_index_list = []
    train_labels = []
    original_labels = {}  # key is the documentIndex, values is the label
    predicted_labels = {}  # key is the documentIndex, values is the label
    complete_labels = []  # indexed by the documentIndex, sameorder from 0 to totalDocuments
    for document_id, document_label in document_to_label.iteritems():
        document_index = docIdToDocIndex[document_id]
        train_index_list.append(document_index)
        train_labels.append(document_label)
        #print document_label
        original_labels[document_index] = document_label

    X = document_collection[train_index_list]
    y = train_labels
    model = None
    if classifier != None:
        model = classifier.fit(X, y)
    for document_index in xrange(0, total_documents):
        if document_index not in train_index_list:
            if classifier != None:
                # normally predict a list of prediction
                # but here we are prediction only 1 iterm
                # so it puts that on a list of one element
                # so we need model.predict(document_collection[document_index])[0]
                predicted_labels[document_index] = model.predict(document_collection[document_index])[0]
                #complete_labels.append(predicted_labels[document_index])
            else:
                predicted_labels[document_index] = 0
                #complete_labels.append(0)  # not relevant any document outside pool
        #else:
        #    complete_labels.append(original_labels[document_index])

    return (topicId, original_labels, predicted_labels)

# this is a normal function which calls the multi-porcessing version
def construct_predicted_qrels(classifier,classifier_name, document_collection, docIdToDocIndex, topic_qrels, data_path, file_name):
    predicted_topic_qrels = {}
    file_exist = True
    for topicId in sorted(topic_qrels.iterkeys()):
        file_complete_path = data_path + file_name + "_" + classifier_name + "_" + str(topicId) + ".pickle"
        if os.path.isfile(file_complete_path) == False:
            file_exist = False
            break
    if file_exist == True:
        print "All file existed. Using those"
    else:

        topic_list = [topicId for topicId in sorted(topic_qrels.iterkeys())]
        #topic_list = ['401']
        num_workers = None
        workers = ProcessPool(num_workers)

        with tqdm(total=len(topic_list)) as pbar:
            partial_construct_predicted_qrels_multi_processsing = partial(construct_predicted_qrels_multi_processsing, classifier = classifier,classifier_name = classifier_name, document_collection = document_collection, docIdToDocIndex = docIdToDocIndex, topic_qrels = topic_qrels)  # prod_x has only one argument x (y is fixed to 10)
            for results in tqdm(workers.imap_unordered(partial_construct_predicted_qrels_multi_processsing, topic_list)):
                # results is a tuple return from the called function
                # then we access the tuples
                topicId = results[0]
                a = results[1]
                b = results[2]
                #c = results[3]
                #predicted_topic_qrels[topicId] = (a,b,c)
                topic_id_complete_qrel = (a,b)
                file_complete_path = data_path + file_name + "_" + classifier_name+"_"+str(topicId) + ".pickle"
                pickle.dump(topic_id_complete_qrel, open(file_complete_path,'wb'))
                pbar.update()
        #pickle.dump(predicted_topic_qrels, open(file_complete_path, 'wb'))
    return predicted_topic_qrels

# topic_seed_info is a dictionary using topicId as a key
# and values is the list of document_index from the whole collection
# similarly topic_original_qrels_in_doc_index is a dictionary using topicId as a key
# and values is the dictionary of document_index from the whole collection and their labels
# this function transform this document index with a base from whole document collection
# to a per topic wise document_index e.g. suppose topic 1 seed info: 10, 100 from whole document collection
# now this function convert this to only index 0,1 because we have only two documents for this topic

def topic_initial_task(topic_original_qrels_in_doc_index, topic_seed_info, document_collection):
    topic_initial_info = {} # topicId is the key
    for topicId in sorted(topic_original_qrels_in_doc_index.iterkeys()):
        per_topic_original_labels_dict = topic_original_qrels_in_doc_index[topicId]
        per_topic_seed_list = topic_seed_info[topicId]
        per_topic_train_index_list = []
        per_topic_train_X = []
        per_topic_train_y = []
        per_topic_doc_index = 0
        per_topic_seed_one_counter = 0.0
        per_topic_seed_zero_counter = 0.0

        # document_index_list is a mapping from per_topic_wise_document_index to collection_wise_document_index
        document_index_list = sorted(per_topic_original_labels_dict.iterkeys())
        per_topic_X = document_collection[document_index_list]

        for document_index in document_index_list:

            doc_label = per_topic_original_labels_dict[document_index]
            per_topic_train_y.append(doc_label)
            if doc_label == 1:
                per_topic_seed_one_counter = per_topic_seed_one_counter + 1
            elif doc_label == 0:
                per_topic_seed_zero_counter = per_topic_seed_zero_counter + 1

            if document_index in per_topic_seed_list:
                # we are keeping the per_topic_doc_index
                # not the document_index in the collection
                per_topic_train_index_list.append(per_topic_doc_index)

            per_topic_doc_index = per_topic_doc_index + 1

        per_topic_y = np.array(per_topic_train_y)
        per_topic_y = per_topic_y.astype('int')

        topic_initial_info[topicId] = (per_topic_X, per_topic_y, per_topic_train_index_list, document_index_list, per_topic_seed_one_counter, per_topic_seed_zero_counter)
    return topic_initial_info


def calculate_accuracy_classifier(topic_X, topic_y, topic_train_index_list, topic_test_index_list, collection_size='qrels'):
    model = None
    if collection_size == 'qrels':
        model = LogisticRegression(solver=small_data_solver, C=small_data_C_parameter)
    y_pred = None
    y_actual = None
    y_pred_hybrid = None
    if len(topic_train_index_list) == len(topic_y):
        y_pred = topic_y
        y_actual = topic_y
        y_pred_hybrid = topic_y

    else:
        model.fit(topic_X[topic_train_index_list], topic_y[topic_train_index_list])
        y_pred = model.predict(topic_X[topic_test_index_list])
        #y_actual = topic_y[topic_test_index_list]

        # this is accuracy of the hybrid human+classifier systems
        y_actual = np.concatenate((topic_y[topic_train_index_list], topic_y[topic_test_index_list]), axis=None)
        y_pred_hybrid = np.concatenate((topic_y[topic_train_index_list], y_pred), axis=None)

    # Accuracy The best performance is 1 with normalize == True and the number of samples with normalize == False
    acc = accuracy_score(y_actual, y_pred_hybrid, normalize=True)
    f1score = f1_score(y_actual, y_pred_hybrid, average='binary')
    precision = precision_score(y_actual, y_pred_hybrid, average='binary')
    recall = recall_score(y_actual, y_pred_hybrid, average='binary')

    #print "inside classifier cal:", len(topic_train_index_list), len(topic_test_index_list), acc, f1score, precision, recall

    return (f1score, precision, recall, y_pred)


def document_selection_task(topicId, topic_initial_info, per_topic_train_index_list, al_protocol, batch_size, collection_size='qrels'):

    per_topic_X, per_topic_y, _, _, per_topic_seed_one_counter, per_topic_seed_zero_counter = topic_initial_info[topicId]

    # this train_index_list will be used to calculate the acuracy of the classifier
    train_index_list = copy.deepcopy(per_topic_train_index_list)
    test_index_list = []
    # it means everything in the train list and we do not need to predict
    # so we do not need any training of the model
    # so return here
    if len(per_topic_train_index_list) == len(per_topic_y):
        return (per_topic_train_index_list, 1.0, 1.0, 1.0, 0, True) # 0 means no new document selected, as f1, precision, recall all reached 1.0, True means this topic is completed
    # print isPredictable.count(1)

    total_documents = len(per_topic_y)
    train_size_controller = len(per_topic_train_index_list)
    size_limit = train_size_controller + batch_size
    number_of_document_selected = batch_size

    # boundary checking
    if size_limit > len(per_topic_y):
        size_limit = len(per_topic_y)
        number_of_document_selected = len(per_topic_y) - len(per_topic_train_index_list)

    per_topic_initial_X_test = []
    per_topic_test_index_dictionary = {}
    test_index_counter = 0

    for train_index in xrange(0, total_documents):
        if train_index not in per_topic_train_index_list:
            per_topic_initial_X_test.append(per_topic_X[train_index])
            per_topic_test_index_dictionary[test_index_counter] = train_index
            test_index_counter = test_index_counter + 1

    predictableSize = len(per_topic_initial_X_test)
    isPredictable = [1] * predictableSize  # initially we will predict all


    # here modeling is utilizing the document selected in previous
    # iteration for training
    # when loopCounter == 0
    # model is utilizing all the seed document collected at the begining
    model = None
    if collection_size == 'qrels':
        model = LogisticRegression(solver=small_data_solver, C=small_data_C_parameter)

    model.fit(per_topic_X[per_topic_train_index_list], per_topic_y[per_topic_train_index_list])

    queueSize = isPredictable.count(1)
    queue = Queue.PriorityQueue(queueSize)

    # these are used for SPL
    randomArray = []

    for counter in xrange(0, predictableSize):
        if isPredictable[counter] == 1:
            # model.predict returns a list of values in so we need index [0] as we
            # have only one element in the list
            y_prob = model.predict_proba(per_topic_initial_X_test[counter])[0]
            val = 0
            if al_protocol == 'CAL':
                val = y_prob[1]
                queue.put(relevance(val, counter))
            elif al_protocol == 'SAL':
                val = calculate_entropy(y_prob[0], y_prob[1])
                queue.put(relevance(val, counter))
            elif al_protocol == 'SPL':
                randomArray.append(counter)



    if al_protocol == 'SPL':
        random.shuffle(randomArray)
        batch_counter = 0
        #for batch_counter in xrange(0, batch_size):
        #    if batch_counter > len(randomArray) - 1:
        #        break
        while True:
            if train_size_controller == size_limit:
                break

            itemIndex = randomArray[batch_counter]
            batch_counter = batch_counter + 1
            isPredictable[itemIndex] = 0
            per_topic_train_index_list.append(per_topic_test_index_dictionary[itemIndex])
            # test_index_list will be used for calculating the accuracy of the classiifer
            test_index_list.append(per_topic_test_index_dictionary[itemIndex])
            train_size_controller = train_size_controller + 1


    else:
        while not queue.empty():
            if train_size_controller == size_limit:
                break
            item = queue.get()
            isPredictable[item.index] = 0  # not predictable

            per_topic_train_index_list.append(per_topic_test_index_dictionary[item.index])
            # test_index_list will be used for calculating the accuracy of the classiifer
            test_index_list.append(per_topic_test_index_dictionary[item.index])
            train_size_controller = train_size_controller + 1

    f1score, precision, recall, _ = calculate_accuracy_classifier(per_topic_X, per_topic_y, train_index_list, test_index_list,collection_size)

    return (per_topic_train_index_list, f1score, precision, recall, number_of_document_selected, False)


def active_learning_multi_processing(topicId, al_protocol, al_classifier, document_collection, topic_seed_info, topic_complete_qrels_address,
                    train_per_centage):
    train_index_list = topic_seed_info[topicId]
    #print topicId
    #print type(train_index_list)
    #print "train_index_list", train_index_list
    #print len(topic_complete_qrels[topicId][0]), len(topic_complete_qrels[topicId][1]), len(topic_complete_qrels[topicId][2])

    topic_complete_qrels = pickle.load(open(topic_complete_qrels_address + topicId + '.pickle', 'rb'))

    original_labels = topic_complete_qrels[0]
    predicted_label = topic_complete_qrels[1]

    original_predicted_merged_dict = {}
    original_labels_list = []
    for k, v in original_labels.iteritems():
        original_predicted_merged_dict[k] = v
        original_labels_list.append(v)
    #exit(0)

    #print "tmp_l1:",original_labels_list.count(1)

    predicted_labels_list = []
    for k, v in predicted_label.iteritems():
        original_predicted_merged_dict[k] = v
        predicted_labels_list.append(v)

    #print "tmp_l2:",predicted_labels_list.count(1)

    #print "sum", original_labels_list.count(1) + predicted_labels_list.count(1)

    original_predicted_merged_list = []
    for k in sorted(original_predicted_merged_dict.iterkeys()):
        #print k, original_predicted_merged_dict[k]
        original_predicted_merged_list.append(original_predicted_merged_dict[k])

    #print "again sum", original_predicted_merged_list.count(1)


    # need to convert y to np.array the Y otherwise Y[train_index_list] does not work directly on a list
    y = np.array(original_predicted_merged_list)  # 2 is complete labels of all documents in document collection
    # type needed because y is an object need and throws error Unknown label type: 'unknown'
    y = y.astype('int')
    #print "numpy sum", np.count_nonzero(y)
    #print y

    #print y.shape
    #print train_index_list
    #print y[train_index_list]

    #exit(0)

    total_documents = len(y)
    total_document_set = set(np.arange(0, total_documents, 1))

    initial_X_test = []
    test_index_dictionary = {}
    test_index_counter = 0

    #print "Starting Test Set Generation:"
    #start = time.time()
    for train_index in xrange(0, total_documents):
        if train_index not in train_index_list:
            initial_X_test.append(document_collection[train_index])
            test_index_dictionary[test_index_counter] = train_index
            test_index_counter = test_index_counter + 1

    #print "Finshed Building Test Set:", time.time() - start

    predictableSize = len(initial_X_test)
    isPredictable = [1] * predictableSize  # initially we will predict all

    # initializing the train_size controller
    train_size_controller = len(train_index_list)
    loopCounter = 1  # loop starts from 1 because 0 is for seed_set
    topic_all_info = {}  # key is the loopCounter

    while True:
        #print "iteration:", loopCounter
        # here modeling is utilizing the document selected in previous
        # iteration for training
        # when loopCounter == 0
        # model is utilizing all the seed document collected at the begining
        if al_classifier == 'LR':
            model = LogisticRegression(solver=large_data_solver, C=large_data_C_parameter, max_iter=200)
        elif al_classifier == 'SVM':
            model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability = True)
        elif al_classifier == 'RF':
            model =  RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)
        elif al_classifier == 'RFN':
            model = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=0)
        elif al_classifier == 'RFN100':
            model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
        elif al_classifier == 'NB':
            model = MultinomialNB()
        elif al_classifier == 'Ada':
            # base model is decision tree
            # logistic regression will not help
            model = AdaBoostClassifier(n_estimators=50,
                                     learning_rate=1)
        elif al_classifier == 'Xgb':
            model = XGBClassifier(random_state=1, learning_rate=0.01)
        elif al_classifier == 'BagLR':
            LRmodel = LogisticRegression(solver=large_data_solver, C=large_data_C_parameter, max_iter=200)
            model = BaggingClassifier(LRmodel, n_estimators = 5, max_samples = 1) # If float, then draw max_samples * X.shape[0] samples. 1 means use all samples
        elif al_classifier == 'BagNB':
            model = BaggingClassifier(MultinomialNB(), n_estimators = 5, max_samples = 0.5) # If float, then draw max_samples * X.shape[0] samples. 1 means use all samples
        elif al_classifier == 'Vot':
            LRmodel = LogisticRegression(solver=large_data_solver, C=large_data_C_parameter, max_iter=200)
            NBmodel = MultinomialNB()
            model = VotingClassifier(estimators=[('lr', LRmodel), ('nb', NBmodel)], voting = 'soft')

        model.fit(document_collection[train_index_list], y[train_index_list])

        test_index_list = list(total_document_set - set(train_index_list))
        pooled_document_count = len(set(train_index_list).intersection(set(original_labels_list)))
        non_pooled_document_count = len(set(train_index_list).intersection(set(predicted_labels_list)))

        y_actual = None
        y_pred = None
        y_pred_all = []

        if isPredictable.count(1) != 0:
            y_pred = model.predict(document_collection[test_index_list])
            start = time.time()
            #print 'Statred y_pred_all'
            y_actual = np.concatenate((y[train_index_list], y[test_index_list]), axis=None)
            y_pred_all = np.concatenate((y[train_index_list], y_pred), axis=None)
            '''
            for doc_index in xrange(0,total_documents):
                if doc_index in train_index_list:
                    y_pred_all.append(y[doc_index])
                else:
                    # result_index in test_set
                    # test_index_list is a list of doc_index
                    # test_Index_list [25, 9, 12]
                    # test_index_list[0] = 25 and its prediction in y_pred[0] --one to one mapping
                    # so find the index of doc_index in test_index_list using
                    pred_index = test_index_list.index(doc_index)
                    y_pred_all.append(y_pred[pred_index])
            '''
            #print "Finsh y_pred_all", time.time() - start

        else: # everything in trainset
            y_pred = y
            y_actual = y
            y_pred_all = y
            test_index_list = train_index_list

        f1score = f1_score(y_actual, y_pred_all, average='binary')
        precision = precision_score(y_actual, y_pred_all, average='binary')
        recall = recall_score(y_actual, y_pred_all, average='binary')

        #print f1score, precision, recall, len(train_index_list), len(test_index_list), len(y_pred_all)

        # save all info using (loopCounter - 1)
        # list should be deep_copy otherwise all will point to final referecne at final iterraion
        topic_all_info[loopCounter - 1] = (topicId, f1score, precision, recall, copy.deepcopy(train_index_list), test_index_list, y_pred, pooled_document_count, non_pooled_document_count)

        # it means everything in the train list and we do not need to predict
        # so we do not need any training of the model
        # so break here
        if isPredictable.count(1) == 0:
            break
        #print isPredictable.count(1)

        queueSize = isPredictable.count(1)
        queue = Queue.PriorityQueue(queueSize)

        # these are used for SPL
        randomArray = []

        for counter in xrange(0, predictableSize):
            if isPredictable[counter] == 1:
                # model.predict returns a list of values in so we need index [0] as we
                # have only one element in the list
                y_prob = model.predict_proba(initial_X_test[counter])[0]
                val = 0
                if al_protocol == 'CAL':
                    val = y_prob[1]
                    queue.put(relevance(val, counter))
                elif al_protocol == 'SAL':
                    val = calculate_entropy(y_prob[0], y_prob[1])
                    queue.put(relevance(val, counter))
                elif al_protocol == 'SPL':
                    randomArray.append(counter)

        size_limit = math.ceil(train_per_centage[loopCounter] * total_documents)

        if al_protocol == 'SPL':
            random.shuffle(randomArray)
            batch_counter = 0
            # for batch_counter in xrange(0, batch_size):
            #    if batch_counter > len(randomArray) - 1:
            #        break
            while True:
                if train_size_controller == size_limit:
                    break

                itemIndex = randomArray[batch_counter]
                isPredictable[itemIndex] = 0
                train_index_list.append(test_index_dictionary[itemIndex])
                train_size_controller = train_size_controller + 1
                batch_counter = batch_counter + 1


        else:
            while not queue.empty():
                if train_size_controller == size_limit:
                    break
                item = queue.get()
                isPredictable[item.index] = 0  # not predictable

                train_index_list.append(test_index_dictionary[item.index])
                train_size_controller = train_size_controller + 1

        loopCounter = loopCounter + 1
    return topic_all_info

def active_learning(topic_list, al_protocol, al_classifier, document_collection, topic_seed_info, topic_complete_qrels_address,train_per_centage, data_path, file_name):
    num_workers = None
    workers = ProcessPool(processes = 1)
    with tqdm(total=len(topic_list)) as pbar:
        partial_active_learning_multi_processing = partial(active_learning_multi_processing, al_protocol=al_protocol, al_classifier = al_classifier, document_collection=document_collection,topic_seed_info=topic_seed_info,topic_complete_qrels_address=topic_complete_qrels_address,train_per_centage=train_per_centage)
        for topic_all_info in tqdm(workers.imap_unordered(partial_active_learning_multi_processing, topic_list)):
            topicId = topic_all_info[0][0] # 0 is the loopCounter Index and 0 is the first tuple
            file_complete_path = data_path + file_name + str(topicId) + ".pickle"
            pickle.dump(topic_all_info, open(file_complete_path, 'wb'))
            pbar.update()

def pick_topic(topic_stats, topic_finished_list, method='min'):
    threshold_accuracy = None
    if method == 'min':
        threshold_accuracy = 1.0
    elif method == 'max':
        threshold_accuracy = 0.0
    picked_topic_id = None
    topicList = []
    for topicId in sorted(topic_stats.iterkeys()):
        topicList.append(topicId)
    #print "under Pick topic:", len(topicList)
    if len(topic_finished_list) == len(topicList):
        #print "returnging now", picked_topic_id
        return (picked_topic_id, threshold_accuracy) # should be None
    for topicId in sorted(topic_stats.iterkeys()):
        if topicId in topic_finished_list:
            continue
        f1score = topic_stats[topicId][1] # 1 index is f1
        if method == 'min':
            if f1score <= threshold_accuracy:
                picked_topic_id = topicId
                threshold_accuracy = f1score
        elif method == 'max':
            if f1score >= threshold_accuracy:
                picked_topic_id = topicId
                threshold_accuracy = f1score
    return (picked_topic_id, threshold_accuracy)

# select a topic using the classifier's accuracy information
def active_topic_selection(datasource, topic_initial_info, batch_size, budget_increment, pick_classifier_method, data_path, file_name):
    total_judged = 0
    topic_stats = {} # key is the topicId values is tuple of (per_topic_train_index_list, f1, precision, recall)
    # initializing the topic stats from topic_initial_info and by calling
    for topicId in sorted(topic_initial_info.iterkeys()):
        per_topic_X, per_topic_y, per_topic_train_index_list, document_index_list, per_topic_seed_one_counter, per_topic_seed_zero_counter = \
        topic_initial_info[topicId]
        total_judged = total_judged + len(per_topic_train_index_list)
        # where calling document_selection_task
        # it will use the initial topic seed and add batch_size ammount of documents to the collection
        # it will increase the per_topic_train_index_list definitely for IS it will become 20
        topic_stats[topicId] = document_selection_task(topicId, topic_initial_info, per_topic_train_index_list, al_protocol, batch_size)


    for topicId in sorted(topic_stats.iterkeys()):
        print topicId, len(topic_stats[topicId][0]), topic_stats[topicId][1]


    last_budget = int(qrelSize[datasource] / 1000) * 1000
    #print "last budget", last_budget
    budget_list = []

    for budget in xrange(2000, last_budget+budget_increment, budget_increment):
        budget_list.append(budget)

    if datasource == 'TREC8':
        for budget in budget_manual_list[datasource]:
            budget_list.append(budget)

    # adding the last dataset point budget_list
    budget_list.append(qrelSize[datasource])
    # sorting the budget_lust
    budget_list = sorted(budget_list)

    topic_finished_list = [] # initially empty but put a topicId when it is complete
    topic_results = {} # key is the topicId and values is a dictionary keyed by budget limit

    for budget_limit in budget_list:
        print "budget now:", budget_limit
        while total_judged < budget_limit:
            picked_topic_id, f1 = pick_topic(topic_stats, topic_finished_list, method = pick_classifier_method)
            print "picked topic:", picked_topic_id, "f1", f1
            # all topic is finished so get exit
            if picked_topic_id == None:
                break
            per_topic_train_index_list = topic_stats[picked_topic_id][0] # 0th tuple is the train_index
            #print "before len train_list", len(per_topic_train_index_list),
            new_per_topic_train_index_list,f1score, precision, recall, number_of_docs_selected, topic_finished = document_selection_task(picked_topic_id, topic_initial_info, per_topic_train_index_list, al_protocol, batch_size)
            #print "new len train_list", len(new_per_topic_train_index_list), "new f1", f1score
            topic_stats[picked_topic_id] = (new_per_topic_train_index_list, f1score, precision, recall, number_of_docs_selected, topic_finished)
            '''
            print "Topic Stats:"
            for topicId in sorted(topic_stats.iterkeys()):
                print topicId, len(topic_stats[topicId][0]), topic_stats[topicId][1]
            '''
            if topic_finished == True:
                topic_finished_list.append(picked_topic_id)
            # we cannot directly put the batch_size to the addition
            # here because sometime topic might have less documents than
            # number specificed in batch_size
            total_judged = total_judged + number_of_docs_selected
        print "total judged now:", total_judged, "with budget ", budget_limit
        for topicId in sorted(topic_stats.iterkeys()):
            train_index_list_under_topic = topic_stats[topicId][0]
            train_X = topic_initial_info[topicId][0]
            train_y = topic_initial_info[topicId][1]
            document_index_list = topic_initial_info[topicId][3]
            total_documents = len(train_y)
            total_document_set = set(np.arange(0, total_documents, 1))
            test_index_list_under_topic = list(total_document_set - set(train_index_list_under_topic))

            f1score, precision, recall, y_pred = calculate_accuracy_classifier(train_X, train_y, train_index_list_under_topic,
                                                                               test_index_list_under_topic, collection_size)

            # converting topic_wise_document_index to collection_wise_document_index for
            # both train_index_list_under_topic and test_index_list_under_topic
            train_index_list = []
            test_index_list = []
            for train_index in train_index_list_under_topic:
                train_index_list.append(document_index_list[train_index])
            for test_index in test_index_list_under_topic:
                test_index_list.append(document_index_list[test_index])


            if topicId in topic_results:
                budget_info = topic_results[topicId] # getting the doctionary
                budget_info[budget_limit] = (topicId, f1score, precision, recall, train_index_list, test_index_list, y_pred)
                topic_results[topicId] = budget_info
            else:
                topic_results[topicId] = {} # creating a dictionary under topic_results[topicId]
                topic_results[topicId][budget_limit] = (topicId, f1score, precision, recall, train_index_list, test_index_list, y_pred)

    for topicId in sorted(topic_results.iterkeys()):
        budget_info = topic_results[topicId]
        # budget_info is a dictionary by budget in budget limit
        file_complete_path = data_path + file_name + str(topicId) + ".pickle"
        pickle.dump(budget_info, open(file_complete_path, 'wb'))


def pseudo_qrel_constructors(list_of_runs_for_qrels_construction, systemRankedDocuments, topic_qrels, pseudo_qrels_file_name, pool_depth):

    pseudo_qrels = {}
    for system_name in list_of_runs_for_qrels_construction:
        if system_name not in systemRankedDocuments:
            #print system_name, "not in system Ranked Documents"
            return None
        documentsFromSystem = systemRankedDocuments[system_name]
        for topicNo, docNo_to_rank in sorted(documentsFromSystem.iteritems()):
            for docNo, docRank in docNo_to_rank.iteritems():
                # if docRank is less than the pool_depth then
                # we would allow the documents in the constructed qrels
                if docRank <= pool_depth:
                    # if that docNo is in original_qrels construc qrels for that
                    if docNo in topic_qrels[topicNo]:
                        # print (system_name, docNo)
                        if topicNo in pseudo_qrels:
                            docNo_label = pseudo_qrels[topicNo]
                            docNo_label[docNo] = topic_qrels[topicNo][docNo]
                            pseudo_qrels[topicNo] = docNo_label
                        else:
                            docNo_label = {}
                            docNo_label[docNo] = topic_qrels[topicNo][docNo]
                            pseudo_qrels[topicNo] = docNo_label

    total_len = 0
    s = ""
    for topicNo, docNo_label in sorted(pseudo_qrels.iteritems()):
        total_len = total_len + len(pseudo_qrels[topicNo])
        # print (len(pseudo_qrels[topicNo]))
        for docNo, label in docNo_label.iteritems():
            s = s + topicNo + " 0 " + docNo + " " + str(label) + "\n"

    #print "Constructing pseudo qrels for:", list_of_runs_for_qrels_construction[0], "size", total_len

    f = open(pseudo_qrels_file_name, "w")
    f.write(s)
    f.close()

    return total_len

# return the number of unique documents (both relevant and non-relevant) return by the systems
# since we are checking the docNo in the topic_qrels so ultimately each runs ranked documents
# is actually controlled by the official pool_depth.
# in another word, we are reading only those documents from a run upto to the pool_depth
def find_all_unique_documents_only_in_set_of_run(list_of_runs, system_returned_documents, topic_qrels, pool_depth):
    topicInfo = {}
    for run in list_of_runs:
        documentsFromSystem = system_returned_documents[run]
        for topicNo, docNo_rank in sorted(documentsFromSystem.iteritems()):
            if topicNo in topicInfo:
                docList = topicInfo[topicNo]
                for docNo in sorted(docNo_rank.iterkeys()):
                    docRank = docNo_rank[docNo]
                    if docRank <= pool_depth:
                        if docNo not in topic_qrels[topicNo]:
                            continue
                        if docNo not in docList:
                            docList.append(docNo)
                topicInfo[topicNo] = docList
            else:
                docList = []
                for docNo in sorted(docNo_rank.iterkeys()):
                    docRank = docNo_rank[docNo]
                    if docRank <= pool_depth:
                        if docNo not in topic_qrels[topicNo]:
                            continue
                        if docNo not in docList:
                            docList.append(docNo)
                topicInfo[topicNo] = docList

    unqiueCount = 0
    for topicNo, docList in sorted(topicInfo.iteritems()):
        unqiueCount = unqiueCount + len(docList)
    return unqiueCount


def find_unique_documents_only_in_set_of_run(list_of_runs, system_returned_documents, topic_qrels, pool_depth):
    topicInfo = {}
    for run in list_of_runs:
        documentsFromSystem = system_returned_documents[run]
        for topicNo, docNo_rank in sorted(documentsFromSystem.iteritems()):
            if topicNo in topicInfo:
                docList = topicInfo[topicNo]
                for docNo in sorted(docNo_rank.iterkeys()):
                    docRank = docNo_rank[docNo]
                    if docRank <= pool_depth:
                        if docNo not in topic_qrels[topicNo]:
                            continue
                        docLabel = topic_qrels[topicNo][docNo]
                        if docLabel == 1:
                            if docNo not in docList:
                                docList.append(docNo)
                topicInfo[topicNo] = docList
            else:
                docList = []
                for docNo in sorted(docNo_rank.iterkeys()):
                    docRank = docNo_rank[docNo]
                    if docRank <= pool_depth:
                        if docNo not in topic_qrels[topicNo]:
                            continue
                        docLabel = topic_qrels[topicNo][docNo]
                        if docLabel == 1:
                            if docNo not in docList:
                                docList.append(docNo)
                topicInfo[topicNo] = docList

    unqiueCount = 0
    for topicNo, docList in sorted(topicInfo.iteritems()):
        unqiueCount = unqiueCount + len(docList)

    return unqiueCount


def find_unique_documents_per_systems(topic_qrels, system_returned_documents):

    topicInfo = {}  # key--TopicID str values -- Dictionary(docNo, systemName)
    for topicNo, docNo_label in sorted(topic_qrels.iteritems()):
        for system_name, documentsFromSystem in sorted(system_returned_documents.iteritems()):
            #print topicNo, system_name
            if topicNo not in documentsFromSystem:
                continue
            documentsFromSystemforTopicNo = documentsFromSystem[topicNo]
            for docNo, label in sorted(docNo_label.iteritems()):
                if docNo in documentsFromSystemforTopicNo:
                    if topicNo in topicInfo:
                        docNo_systemName = topicInfo[topicNo]
                        if docNo in docNo_systemName:
                            systemNameList = docNo_systemName[docNo]
                            systemNameList.append(system_name)
                            docNo_systemName[docNo] = systemNameList
                        else:
                            systemNameList = []
                            systemNameList.append(system_name)
                            docNo_systemName[docNo] = systemNameList
                        topicInfo[topicNo] = docNo_systemName

                    else:
                        docNo_systemName = {}
                        systemNameList = []
                        systemNameList.append(system_name)
                        docNo_systemName[docNo] = systemNameList
                        topicInfo[topicNo] = docNo_systemName

    uniqueDocumentsFromSystems = {}
    uniqueDocumentsFromSystemsCount = {}
    for system_name, _ in sorted(system_returned_documents.iteritems()):
        uniqueDocumentsFromSystemsCount[system_name] = 0
    for topicNo, docNo_systemName in sorted(topicInfo.iteritems()):
        for docNo, system_name_list in sorted(docNo_systemName.iteritems()):
            #print topicNo, docNo, len(system_name_list)
            docLabel = topic_qrels[topicNo][docNo]
            if len(system_name_list) == 1 and docLabel == 1: # this is an unique document returned by this system
                uniqueDocumentsFromSystemsCount[system_name_list[0]] = uniqueDocumentsFromSystemsCount[system_name_list[0]] + 1
                if system_name_list[0] in uniqueDocumentsFromSystems:
                    topicToDocuments = uniqueDocumentsFromSystems[system_name_list[0]]
                    if topicNo in topicToDocuments:
                        documentList = topicToDocuments[topicNo]
                        documentList.append(docNo)
                        topicToDocuments[topicNo] = documentList
                    else:
                        documentList = []
                        documentList.append(docNo)
                        topicToDocuments[topicNo] = documentList
                    uniqueDocumentsFromSystems[system_name_list[0]] = topicToDocuments
                else:
                    documentList = []
                    documentList.append(docNo)
                    topicToDocuments = {}
                    topicToDocuments[topicNo] = documentList
                    uniqueDocumentsFromSystems[system_name_list[0]] = topicToDocuments
    total_documents = 0
    for system_name, count in sorted(uniqueDocumentsFromSystemsCount.iteritems()):
        print system_name, count
        total_documents = total_documents + count

    print total_documents
    return uniqueDocumentsFromSystems, uniqueDocumentsFromSystemsCount

# running experiments using excluded systems not exlcuded groups
def excluding_system_runs_experiment(data_path, datasource):
    runs_group_list = group_list[datasource]
    # 25% of the groups are excluded
    number_of_groups_to_exclude = int(math.ceil(len(runs_group_list) * 0.25))

    excluded_groups_file_name = data_path + "excluded_groups_" + datasource + ".pickle"
    excluded_systems = {}

    if os.path.exists(excluded_groups_file_name):
        excluded_systems = pickle.load(open(excluded_groups_file_name, "rb"))
        print ("Excluded system list exist")
    else:
        print ("Excluded system list NOT exist")
        random.seed(9001)
        for i in xrange(0, 5, 1):
            excluded_systems_list = []
            random_group_numbers = random.sample(xrange(len(runs_group_list)), number_of_groups_to_exclude)
            for group_number in random_group_numbers:
                system_list = runs_group_list[group_number + 1]  # because I started gropu number form 1
                for system_run in system_list:
                    excluded_systems_list.append(system_run)
            excluded_systems[i] = excluded_systems_list

        pickle.dump(excluded_systems, open(excluded_groups_file_name, "wb"))

    excluded_systems_index_list = [excluded_systems_index_list_value]
    print "excluded_systems_index_list", excluded_systems_index_list

    system_numbers_list_labels = None
    excluded_systems_tau_list = {}
    excluded_systems_drop_list = {}
    excluded_systems_unique_doc_count_list = {}
    for excluded_systems_index in excluded_systems_index_list:
        exlcuded_system_names = excluded_systems[excluded_systems_index]

        remaining_system_names = []
        for system_name in systemNameList[datasource]:
            if system_name not in exlcuded_system_names:
                remaining_system_names.append(system_name)

        system_numbers_tau_list = {}
        system_numbers_drop_list = {}
        system_numbers_unique_doc_count_list = {}

        system_numbers_list_1 = list(xrange(1, 10, 2))
        system_numbers_list_2 = list(xrange(10, len(remaining_system_names), 10))

        system_numbers_list_final = sorted(system_numbers_list_1 + system_numbers_list_2)
        system_numbers_list_labels = system_numbers_list_final
        for system_numbers in system_numbers_list_final:
            number_shuffle_tau_list = []
            number_shuffle_drop_list = []
            number_shuffle_unique_doc_count_list = []

            for number_shuffle in xrange(0, 5):
                # we need to shuffle the system runs list 5 times
                # so e.g., for run 1--> system 1, 3 ,5 added
                # then for run 2--> system 3, 5, 1 added
                # sampling system_numbers of systems
                my_randoms = random.sample(xrange(len(remaining_system_names)), system_numbers)
                print my_randoms
                system_run_list_names = []
                for number in my_randoms:
                    system_run_list_names.append(remaining_system_names[number])

                pseudo_qrels_file_name = data_path + datasource + "_pseudo_qrels_" + str(
                    excluded_systems_index) + "_" + str(system_numbers) + "_" + str(number_shuffle) + ".txt"

                total_len = pseudo_qrel_constructors(system_run_list_names, systemRankedDocuments,
                                                     topic_qrels,
                                                     pseudo_qrels_file_name)

                original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
                    exlcuded_system_names, systemAddress[datasource], qrelAddress[datasource], rankMetric)
                original_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_original_' + rankMetric + '.pickle'
                # pickle.dump(original_system_metric_value, open(original_system_metric_value_file_name, 'wb'))
                print system_run_list_names
                relevanceJudgementAddress = pseudo_qrels_file_name

                predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
                    exlcuded_system_names, systemAddress[datasource], relevanceJudgementAddress, rankMetric)
                # predicted_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_predicted_' + rankMetric + '_' + str(
                #    i) + '.pickle'
                # pickle.dump(predicted_system_metric_value, open(predicted_system_metric_value_file_name, 'wb'))
                tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)
                number_shuffle_tau_list.append(tau)

                uniqueCount = find_unique_documents_only_in_set_of_run(system_run_list_names,
                                                                       systemRankedDocuments, topic_qrels)

                max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                                      predicted_system_metric_value_list)
                number_shuffle_drop_list.append(max_drop)
                number_shuffle_unique_doc_count_list.append(uniqueCount)

                print excluded_systems_index, system_numbers, number_shuffle, tau, max_drop, uniqueCount

            system_numbers_tau_list[system_numbers] = number_shuffle_tau_list
            system_numbers_drop_list[system_numbers] = number_shuffle_drop_list
            system_numbers_unique_doc_count_list[system_numbers] = number_shuffle_unique_doc_count_list
        excluded_systems_tau_list[excluded_systems_index] = system_numbers_tau_list
        excluded_systems_drop_list[excluded_systems_index] = system_numbers_drop_list
        excluded_systems_unique_doc_count_list[excluded_systems_index] = system_numbers_unique_doc_count_list

    excluded_systems_tau_list_file_name = data_path + datasource + "_pseudo_qrels_all_taus.pickle"
    print excluded_systems_tau_list_file_name
    pickle.dump(excluded_systems_tau_list, open(excluded_systems_tau_list_file_name, 'wb'))

    excluded_systems_tau_drop_uniqueDocs_file_name = data_path + datasource + "_pseudo_qrels_all_taus_drop_unique_counts_" + str(
        excluded_systems_index_list[0]) + ".pickle"
    print excluded_systems_tau_drop_uniqueDocs_file_name
    excluded_systems_tau_drop_uniqueDocs_object = [excluded_systems_tau_list, excluded_systems_drop_list,
                                                   excluded_systems_unique_doc_count_list, system_numbers_list_labels]
    pickle.dump(excluded_systems_tau_drop_uniqueDocs_object, open(excluded_systems_tau_drop_uniqueDocs_file_name, 'wb'))



# running experiments using excluded exlcuded groups
def excluding_groups_experiment(data_path, datasource, excluded_systems_index_list_value, systemRankedDocuments,
                                                     topic_qrels, rankMetric, grp_start_number, grp_end_number, onlyautomatic, pool_depth_to_use):
    print "Rank Metric Used:", rankMetric
    runs_group_list = group_list[datasource]
    # 25% of the groups are excluded
    number_of_groups_to_exclude = int(math.ceil(len(runs_group_list) * 0.25))

    excluded_groups_file_name = data_path + "excluded_groups_" + datasource + str(onlyautomatic) + ".pickle"
    excluded_systems = {}

    if os.path.exists(excluded_groups_file_name):
        excluded_systems = pickle.load(open(excluded_groups_file_name, "rb"))
        print ("Excluded system list exist")
    else:
        print ("Excluded system list NOT exist")
        random.seed(9001)
        for i in xrange(0, 5, 1):
            excluded_systems_list = []
            random_group_numbers = random.sample(xrange(len(runs_group_list)), number_of_groups_to_exclude)
            for group_number in random_group_numbers:
                system_list = runs_group_list[group_number + 1]  # because I started gropu number form 1
                for system_run in system_list:
                    excluded_systems_list.append(system_run)
            excluded_systems[i] = excluded_systems_list

        pickle.dump(excluded_systems, open(excluded_groups_file_name, "wb"))

    excluded_systems_index_list = [excluded_systems_index_list_value]
    print "excluded_systems_index_list", excluded_systems_index_list

    system_numbers_list_labels = None
    excluded_systems_tau_list = {}
    excluded_systems_drop_list = {}
    excluded_systems_unique_doc_count_list = {}
    exlcuded_systems_tau_ap_list = {}

    for excluded_systems_index in excluded_systems_index_list:
        exlcuded_system_names = excluded_systems[excluded_systems_index]

        remaining_system_names = []
        remaining_group_numbers = []
        for system_name in systemNameList[datasource]:
            if system_name not in exlcuded_system_names:
                remaining_system_names.append(system_name)
                grp_no = find_group_number(datasource, system_name)
                if grp_no not in remaining_group_numbers:
                    remaining_group_numbers.append(grp_no)

        remaining_group_numbers = sorted(remaining_group_numbers)
        print "remaining group number", remaining_group_numbers

        system_numbers_tau_list = {}
        system_numbers_tau_ap_list = {}
        system_numbers_drop_list = {}
        system_numbers_unique_doc_count_list = {}

        group_number_list = sorted(list(xrange(group_considered_start[datasource], len(remaining_group_numbers)+1, group_considered_step[datasource])))
        system_numbers_list_labels = group_number_list
        print "group_number_list", group_number_list
        # here system_numbers variable actually contains group_numbers
        for system_numbers in group_number_list:
            if system_numbers >= grp_start_number and system_numbers <= grp_end_number:

                number_shuffle_tau_list = []
                number_shuffle_tau_ap_list = []
                number_shuffle_drop_list = []
                number_shuffle_unique_doc_count_list = []

                for number_shuffle in xrange(0, 5):
                    # we need to shuffle the system runs list 5 times
                    # so e.g., for run 1--> system 1, 3 ,5 added
                    # then for run 2--> system 3, 5, 1 added
                    # sampling system_numbers of systems
                    my_randoms = random.sample(xrange(len(remaining_group_numbers)), system_numbers)
                    #print "randomly selected group_number_index", my_randoms
                    system_run_list_names = []
                    random_group_numbers_list = []
                    for number in my_randoms:
                        random_grp_no = remaining_group_numbers[number]
                        random_group_numbers_list.append(random_grp_no)
                        system_name_list_for_this_group = group_list[datasource][random_grp_no]
                        for run_name in system_name_list_for_this_group:
                            system_run_list_names.append(run_name)

                    print "randomly selected group number", random_group_numbers_list

                    pseudo_qrels_file_name = data_path + datasource + "grp_start_number_" + str(grp_start_number) + "_pseudo_qrels_group_format_" + str(
                        excluded_systems_index) + "_" + str(system_numbers) + "_" + str(number_shuffle) + "_" + str(onlyautomatic) + "_" + rankMetric + ".txt"

                    total_len = pseudo_qrel_constructors(system_run_list_names, systemRankedDocuments,
                                                         topic_qrels,
                                                         pseudo_qrels_file_name)

                    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
                        exlcuded_system_names, systemAddress[datasource], qrelAddress[datasource], rankMetric)
                    original_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_original_' + rankMetric + '.pickle'
                    # pickle.dump(original_system_metric_value, open(original_system_metric_value_file_name, 'wb'))
                    print system_run_list_names
                    relevanceJudgementAddress = pseudo_qrels_file_name

                    predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
                        exlcuded_system_names, systemAddress[datasource], relevanceJudgementAddress, rankMetric)
                    # predicted_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_predicted_' + rankMetric + '_' + str(
                    #    i) + '.pickle'
                    # pickle.dump(predicted_system_metric_value, open(predicted_system_metric_value_file_name, 'wb'))
                    tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)
                    number_shuffle_tau_list.append(tau)

                    tau_ap = tau_ap_mine(original_system_metric_value_list,predicted_system_metric_value_list)
                    number_shuffle_tau_ap_list.append(tau_ap)

                    uniqueCount = find_unique_documents_only_in_set_of_run(system_run_list_names,
                                                                           systemRankedDocuments, topic_qrels)

                    max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                                          predicted_system_metric_value_list)
                    number_shuffle_drop_list.append(max_drop)
                    number_shuffle_unique_doc_count_list.append(uniqueCount)

                    print excluded_systems_index, system_numbers, number_shuffle, tau, tau_ap, max_drop, uniqueCount

                    # removing the pseudo_qrels
                    if os.path.exists(pseudo_qrels_file_name):
                        os.remove(pseudo_qrels_file_name)

                system_numbers_tau_list[system_numbers] = number_shuffle_tau_list
                system_numbers_tau_ap_list[system_numbers] = number_shuffle_tau_ap_list

                system_numbers_drop_list[system_numbers] = number_shuffle_drop_list
                system_numbers_unique_doc_count_list[system_numbers] = number_shuffle_unique_doc_count_list
        excluded_systems_tau_list[excluded_systems_index] = system_numbers_tau_list
        exlcuded_systems_tau_ap_list[excluded_systems_index] = system_numbers_tau_ap_list
        excluded_systems_drop_list[excluded_systems_index] = system_numbers_drop_list
        excluded_systems_unique_doc_count_list[excluded_systems_index] = system_numbers_unique_doc_count_list

    excluded_systems_tau_list_file_name = data_path + datasource + "grp_start_number_" + str(grp_start_number) +"_pseudo_qrels_all_taus_group_format_" + str(onlyautomatic) + "_" + rankMetric + "_" + ".pickle"
    print excluded_systems_tau_list_file_name
    pickle.dump(excluded_systems_tau_list, open(excluded_systems_tau_list_file_name, 'wb'))

    excluded_systems_tau_drop_uniqueDocs_file_name = data_path + datasource + "grp_start_number_" + str(grp_start_number) + "_pseudo_qrels_all_taus_drop_unique_counts_group_format_" + str(onlyautomatic) + "_" + rankMetric + "_" + str(
        excluded_systems_index_list[0]) + ".pickle"
    print excluded_systems_tau_drop_uniqueDocs_file_name
    excluded_systems_tau_drop_uniqueDocs_object = [excluded_systems_tau_list, exlcuded_systems_tau_ap_list, excluded_systems_drop_list,
                                                   excluded_systems_unique_doc_count_list, system_numbers_list_labels]
    pickle.dump(excluded_systems_tau_drop_uniqueDocs_object, open(excluded_systems_tau_drop_uniqueDocs_file_name, 'wb'))



def leave_one_out_experiments(data_path, datasource, sample_number_considered, systemRankedDocuments,
                                                     topic_qrels, rankMetric, grp_start_number, grp_end_number, onlyautomatic, pool_depth_to_use):
    # change the group_list variable to a local name so it does not affect the variable in global_defintion.py file
    # we need it for only group_list and group_list_without_manual_runs variables

    group_samples_file_name = data_path + "group_samples_"+ str(onlyautomatic)+ "_" + datasource + ".pickle"
    groups_considered = {}

    group_list_datasource = []

    if onlyautomatic == 0: # use all manual and automatic runs
        print "Using both automatic and manual runs"
        group_list_datasource = group_list[datasource]
    elif onlyautomatic == 1: # use only automatic runs
        print "Using only automatic runs"
        group_list_datasource  = group_list_without_manual_runs[datasource]

    if os.path.exists(group_samples_file_name):
        groups_considered = pickle.load(open(group_samples_file_name, "rb"))
        print ("Group Sample File list exist")
    else:
        print ("Group Sample File Does NOT exist")
        total_number_of_groups = len(group_list_datasource)
        random.seed(9001)
        for number_of_groups in xrange(group_considered_start[datasource], total_number_of_groups + 1, group_considered_step[datasource]):
            sample_numbers = {}
            for sample_num in xrange(0, 5, 1):
                random_group_numbers = random.sample(xrange(total_number_of_groups), number_of_groups)
                sample_numbers[sample_num] = random_group_numbers
            groups_considered[number_of_groups] = sample_numbers

        # when we pick all groups we do not need to sample N times
        # so we have only one sample
        #if datasource != 'TREC7': # because for TREC7 we have both 40 and 41 and for 41 we have bufferoverflow error
        #    sample_numbers = {}
        #    sample_numbers[0] = list(xrange(total_number_of_groups))
        #    groups_considered[total_number_of_groups] = sample_numbers
        #    pickle.dump(groups_considered, open(group_samples_file_name, "wb"))


    group_considered_tau = {}
    group_considered_tau_ap = {}
    group_considered_drop = {}
    group_considered_unique = {}
    group_considered_unique_all = {}

    # group number is indexed with 1
    for group_number, sample_numbers in sorted(groups_considered.iteritems()):
        print "Before", group_number, sample_numbers
        if group_number >= grp_start_number and group_number <= grp_end_number:
            sample_numbers_tau = {}
            sample_numbers_tau_ap = {}
            sample_numbers_unique = {}
            sample_numbers_unique_all = {}

            sample_numbers_drop = {}

            #if group_number == len(group_list[datasource]):
            #    sample_number_considered = 0

            for sample_num, group_number_list in sorted(sample_numbers.iteritems()):
                if sample_num != sample_number_considered:
                    continue
                temp_groups = {}
                for group_num in group_number_list:
                    temp_groups[group_num + 1] = group_list_datasource[group_num + 1] # because I stared gorup index form 1

                except_group_tau = []
                except_group_tau_ap = []
                except_group_max_drop = []
                except_group_unique_set_count = []
                except_group_unique_set_count_all = []

                list_of_runs_for_original_qrels_construction = []
                for i in sorted(temp_groups.iterkeys()):
                    for run in group_list_datasource[i]:
                        list_of_runs_for_original_qrels_construction.append(run)

                pseudo_original_qrels_file_name = data_path + datasource + "grp_start_number_" + str(grp_start_number) + "_original_pseudo_qrels_system_numbers_of_groups_" + str(group_number) + "_sample_number_" + str(sample_num) + "_" + rankMetric + "_" + str(onlyautomatic) + "_" + str(pool_depth_to_use) + ".txt"
                total_len = pseudo_qrel_constructors(list_of_runs_for_original_qrels_construction, systemRankedDocuments,
                                                     topic_qrels,
                                                     pseudo_original_qrels_file_name, pool_depth_to_use)

                original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
                    list_of_runs_for_original_qrels_construction, systemAddress[datasource], pseudo_original_qrels_file_name, rankMetric)

                for i in sorted(temp_groups.iterkeys()):
                    list_of_runs_for_qrels_construction = []
                    excluded_groups = temp_groups[i]
                    for j in sorted(temp_groups.iterkeys()):
                        if j == i:
                            continue
                        for run in group_list_datasource[j]:
                            list_of_runs_for_qrels_construction.append(run)

                    unique_dcouments_count = 0
                    max_unique_document = 0
                    for run in list_of_runs_for_qrels_construction:
                        unique_dcouments_count = unique_dcouments_count + uniqueDocumentsFromSystemsCount[run]
                        max_unique_document = max(max_unique_document, uniqueDocumentsFromSystemsCount[run])

                    pseudo_qrels_file_name = data_path + datasource + "grp_start_number_" + str(grp_start_number) + "_pseudo_qrels_system_except_group_" + str(i)+"_numbers_of_groups_" + str(group_number) +"_sample_number_" + str(sample_num) +"_" + rankMetric + "_" + str(onlyautomatic) + "_" + str(pool_depth_to_use) + ".txt"
                    total_len = pseudo_qrel_constructors(list_of_runs_for_qrels_construction, systemRankedDocuments,
                                                         topic_qrels,
                                                         pseudo_qrels_file_name, pool_depth_to_use)


                    relevanceJudgementAddress = pseudo_qrels_file_name

                    predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
                        list_of_runs_for_original_qrels_construction, systemAddress[datasource], relevanceJudgementAddress, rankMetric)

                    tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

                    tau_ap = tau_ap_mine(original_system_metric_value_list, predicted_system_metric_value_list)

                    uniqueCount = find_unique_documents_only_in_set_of_run(list_of_runs_for_qrels_construction,
                                                                           systemRankedDocuments, topic_qrels, pool_depth_to_use)

                    uniqueAllCount = find_all_unique_documents_only_in_set_of_run(list_of_runs_for_qrels_construction,
                                                                           systemRankedDocuments, topic_qrels, pool_depth_to_use)

                    max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                                          predicted_system_metric_value_list)

                    except_group_max_drop.append(max_drop)
                    except_group_tau.append(tau)
                    except_group_tau_ap.append(tau_ap)
                    except_group_unique_set_count.append(uniqueCount)
                    except_group_unique_set_count_all.append(uniqueAllCount)

                    print group_number, sample_num, i, tau, tau_ap, max_drop, uniqueCount, uniqueAllCount

                    # removing the file
                    if os.path.exists(pseudo_qrels_file_name):
                        print "removing file:", pseudo_qrels_file_name
                        os.remove(pseudo_qrels_file_name)

                if os.path.exists(pseudo_original_qrels_file_name):
                    print "Removing file:", pseudo_original_qrels_file_name
                    os.remove(pseudo_original_qrels_file_name)

                sample_numbers_tau[sample_num] = except_group_tau
                sample_numbers_tau_ap[sample_num] = except_group_tau_ap
                sample_numbers_drop[sample_num] = except_group_max_drop
                sample_numbers_unique[sample_num] = except_group_unique_set_count
                sample_numbers_unique_all[sample_num] = except_group_unique_set_count_all



            group_considered_tau[group_number] = sample_numbers_tau
            group_considered_tau_ap[group_number] = sample_numbers_tau_ap
            group_considered_drop[group_number] = sample_numbers_drop
            group_considered_unique[group_number] = sample_numbers_unique
            group_considered_unique_all[group_number] = sample_numbers_unique_all

    # writing all values
    print "Writing START PICKLE here"
    group_consider_values = [group_considered_tau, group_considered_tau_ap, group_considered_drop, group_considered_unique, group_considered_unique_all]
    group_considered_file_name = data_path + "grp_start_number_" + str(grp_start_number) + "_group_considered_sample_number_"+str(sample_number_considered)+ "_" + datasource+"_" + rankMetric + "_" + str(onlyautomatic) + "_" + str(pool_depth_to_use) + ".pickle"
    pickle.dump(group_consider_values, open(group_considered_file_name, "wb"))
    print "Writing DONE in PICKLE here", group_considered_file_name

#########################################################################
# main script
# create all directories manually before running otherwise multi-processing will create lock condition
# for creating files
# rankMetric
if __name__ == '__main__':
    datasource = sys.argv[1]  # can be 'TREC8','gov2', 'WT2013','WT2014'
    al_protocol = sys.argv[2]  # 'SAL', 'CAL', # SPL is not there yet
    seed_selection_type = sys.argv[3] # 'IS' only
    classifier_name = sys.argv[4] # "LR", "NR"--> means non-relevant all
    collection_size = sys.argv[5] # 'all', 'qrels' qrels --> means consider documents inseide qrels only
    al_classifier = sys.argv[6] # SVM, RF, NB and LR
    start_top = int(sys.argv[7])
    end_top = int(sys.argv[8])
    rankMetric = sys.argv[9]
    excluded_systems_index_list_value = int(sys.argv[10])
    sample_number_considered = int(sys.argv[10])
    grp_start_number = int(sys.argv[11])
    grp_end_number = int(sys.argv[12])
    onlyautomatic = int(sys.argv[13])
    pool_depth_to_use = int(sys.argv[14]) # the pool depth to use to construct the qrels


    print "datasource:", datasource, "rankmetric:", rankMetric, "exclude:", excluded_systems_index_list_value, "grp_start:", grp_start_number, "grp_end:", grp_end_number, "pool_depth:", pool_depth_to_use

    source_file_path =  base_address + datasource + "/"
    data_path = base_address + datasource + "/result/"
    if collection_size == 'qrels':
        source_file_path =  base_address + datasource + "/sparseTRECqrels/"
        data_path = base_address + datasource + "/sparseTRECqrels/" + "result/"

    data_path = data_path + al_classifier + "/"
    print "source_file_path", source_file_path
    print "data_path", data_path

    #topicData = TRECTopics(datasource, start_topic[datasource], end_topic[datasource])

    topicData = TRECTopics(datasource, start_top, end_top)

    # topic_qrels is a dictionary
    # where topic Id is a string for key and docList is the list of documents with labels

    # load relevance judgement per topic wise
    # return a dictionary of topic_to_docList key is topicid from TREC and doclist is the list of document related to that topic

    topic_qrels = topicData.qrelsReader(qrelAddress[datasource], data_path, topic_original_qrels_filename)

    '''
    for topicId in xrange(start_top, end_top):
        for docId, docLabel in topic_qrels[str(topicId)].iteritems():
            print (topicId, docId, docLabel)
    '''

    #manual_auto_runs_list = [f for f in listdir(systemAddress['TREC8']) if isfile(join(systemAddress['TREC8'], f))]
    #system_runs_TREC8_list = sorted(manual_auto_runs_list)

    # return a dictionary
    # where key is the topicid (str)
    # values is a dictionary of (documentNo, rank from TREC)

    systemData = systemReader(datasource, start_top, end_top)


    systemRankedDocuments = {}
    system_ranked_file_path = data_path + datasource + '_systems_ranked_documents.pickle'

    if os.path.exists(system_ranked_file_path):
        print ("Systems Ranked Document Pickle Exists")
        systemRankedDocuments = pickle.load(open(system_ranked_file_path, 'rb'))

    else:
        print ("Systems Ranked Document Does not Pickle Exists")
        for system_name in systemNameList[datasource]:
            systemRankedDocuments[system_name] = systemData.rankedDocumentFromSystem(systemAddress[datasource], system_name, pool_depth[datasource])
            print (system_name)

        print (system_ranked_file_path)
        pickle.dump(systemRankedDocuments, open(system_ranked_file_path, 'wb'))

    uniqueDocumentsFromSystems, uniqueDocumentsFromSystemsCount = find_unique_documents_per_systems(topic_qrels, systemRankedDocuments)

    #excluding_groups_experiment(data_path, datasource, excluded_systems_index_list_value, systemRankedDocuments,topic_qrels, rankMetric, grp_start_number, grp_end_number, onlyautomatic, pool_depth_to_use)

    leave_one_out_experiments(data_path, datasource, sample_number_considered, systemRankedDocuments,topic_qrels, rankMetric, grp_start_number, grp_end_number, onlyautomatic, pool_depth_to_use)

    exit(0)



    '''
    # Mucahid's idea
    scatter_plot_x = []
    scatter_plot_y = []
    scatter_plot_tau_as_y = []
    
    for i in group_list['TREC8'].iterkeys():
        list_of_runs_for_qrels_construction = []
        excluded_groups = group_list['TREC8'][i]
        for j in xrange(1, len(group_list['TREC8'])+1):
            if j == i:
                continue
            for run in group_list['TREC8'][j]:
                #if run not in manual_run_list['TREC8']: # considering only manual run
                list_of_runs_for_qrels_construction.append(run)

        unique_dcouments_count = 0
        max_unique_document = 0
        for run in list_of_runs_for_qrels_construction:
            unique_dcouments_count = unique_dcouments_count + uniqueDocumentsFromSystemsCount[run]
            max_unique_document = max(max_unique_document, uniqueDocumentsFromSystemsCount[run])
        pseudo_qrels_file_name = data_path + datasource + "_pseudo_qrels_system_except_group_" + str(i) + ".txt"
        total_len = pseudo_qrel_constructors(list_of_runs_for_qrels_construction, systemRankedDocuments, topic_qrels,
                                             pseudo_qrels_file_name)

        original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
            system_runs_TREC8_list, systemAddress[datasource], qrelAddress[datasource], rankMetric)
        original_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_original_' + rankMetric + '.pickle'
        # pickle.dump(original_system_metric_value, open(original_system_metric_value_file_name, 'wb'))

        relevanceJudgementAddress = pseudo_qrels_file_name

        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
            system_runs_TREC8_list, systemAddress[datasource], relevanceJudgementAddress, rankMetric)
        # predicted_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_predicted_' + rankMetric + '_' + str(
        #    i) + '.pickle'
        # pickle.dump(predicted_system_metric_value, open(predicted_system_metric_value_file_name, 'wb'))
        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

        uniqueCount = find_unique_documents_only_in_set_of_run(list_of_runs_for_qrels_construction, systemRankedDocuments, topic_qrels)
        except_group_tau.append(tau)
        except_group_unique_documents.append(uniqueCount)
        except_group_max_unique_documents.append(max_unique_document)
        except_group_unique_set_count.append(uniqueCount)

        for run in excluded_groups:
            map_run_original = original_system_metric_value[run]
            map_run_excluded = predicted_system_metric_value[run]
            scatter_plot_x.append(uniqueCount)
            scatter_plot_y.append(abs(map_run_original - map_run_excluded))
            scatter_plot_tau_as_y.append(tau)

        print tau, uniqueCount, unique_dcouments_count, max_unique_document

    group_vs_tau = (except_group_unique_set_count, except_group_max_unique_documents, except_group_tau, scatter_plot_x, scatter_plot_y, scatter_plot_tau_as_y)
    group_vs_tau_file_name = data_path + "group_vs_tau_uniue_set_all.pickle"

    pickle.dump(group_vs_tau, open(group_vs_tau_file_name, 'wb'))
    
    group_vs_tau_file_name = data_path + "group_vs_tau_uniue_set_all.pickle"

    group_vs_tau = pickle.load(open(group_vs_tau_file_name, 'rb'))
    except_group_unique_set_count = group_vs_tau[0]
    except_group_max_unique_documents = group_vs_tau[1]
    except_group_tau = group_vs_tau[2]
    scatter_plot_x = group_vs_tau[3]
    scatter_plot_y = group_vs_tau[4]
    scatter_plot_tau_as_y = group_vs_tau[5]

    import matplotlib.pyplot as plt

    plt.scatter(except_group_unique_set_count, except_group_tau)
    plt.title('Total # of unique relevant documents vs. tau')
    plt.xlabel('Number of unique relevant documents')
    plt.ylabel('tau correlation')

    plt.savefig(data_path + 'tau_plots_vs_group_uniue_set_all.png', bbox_inches='tight')
    plt.clf()

    plt.scatter(scatter_plot_x, scatter_plot_y)
    plt.title('map differences of the excluded groups vs Total # of unique relevant documents')
    plt.xlabel('Number of unique relevant documents')
    plt.ylabel('map difference')

    plt.savefig(data_path + 'map_difference_vs_group_uniue_set_all.png', bbox_inches='tight')
    plt.clf()

    from scipy.stats import pearsonr

    pr = pearsonr(scatter_plot_tau_as_y, scatter_plot_y)
    plt.scatter(scatter_plot_tau_as_y, scatter_plot_y)
    plt.title('map differences of the excluded groups vs Total # of unique relevant documents \n pearson correlation:' + str(pr))
    plt.xlabel('tau correlation')
    plt.ylabel('map difference')

    plt.savefig(data_path + 'map_difference_vs_tau_all.png', bbox_inches='tight')
    plt.clf()

    exit(0)
    '''


    '''
    # this is code for constructing pseduo_qrels for only using one system at a time
    # and rank all systems including this system
    
    manual_runs_tau_list = []
    for run in manual_run_list['TREC8']:
        #pseudo_qrels_file_name = data_path + datasource + "_pseudo_qrels_" + str(excluded_systems_index) + "_" + str(
        #    number_shuffle) + "_" + str(j) + ".txt"
        if run not in systemRankedDocuments:
            continue
        list_of_runs_for_qrels_construction = []
        list_of_runs_for_qrels_construction.append(run)
        pseudo_qrels_file_name = data_path + datasource + "_pseudo_qrels_system_" + run + ".txt"
        total_len = pseudo_qrel_constructors(list_of_runs_for_qrels_construction, systemRankedDocuments, topic_qrels,
                                 pseudo_qrels_file_name)

        original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
            system_runs_TREC8_list, systemAddress[datasource], qrelAddress[datasource], rankMetric)
        original_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_original_' + rankMetric + '.pickle'
        #pickle.dump(original_system_metric_value, open(original_system_metric_value_file_name, 'wb'))

        relevanceJudgementAddress = pseudo_qrels_file_name

        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
            system_runs_TREC8_list, systemAddress[datasource], relevanceJudgementAddress, rankMetric)
        #predicted_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_predicted_' + rankMetric + '_' + str(
        #    i) + '.pickle'
        #pickle.dump(predicted_system_metric_value, open(predicted_system_metric_value_file_name, 'wb'))
        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

        drop_in_rank_list, delta_in_score_list = drop_calculator(original_system_metric_value_list, predicted_system_metric_value_list)

        print run, total_len, tau
        manual_runs_tau_list.append((run, total_len, tau))

    manual_runs_tau_list_file_name = data_path + datasource + "_manual_runs_taus.pickle"
    pickle.dump(manual_runs_tau_list, open(manual_runs_tau_list_file_name, 'wb'))

    manual_runs_tau_list_file_name = data_path + datasource + "_manual_runs_taus.pickle"
    manual_runs_tau_list = pickle.load(open(manual_runs_tau_list_file_name, "rb"))

    for list_item in manual_runs_tau_list:
        print list_item[0], ",", list_item[1], ",", list_item[2]

    exit(0)
    '''
    excluding_system_runs_experiment(data_path, datasource)

    exit(0)

    excluded_systems_tau_list_file_name = data_path + datasource + "_pseudo_qrels_all_taus.pickle"
    print excluded_systems_tau_list_file_name
    excluded_systems_tau_list = pickle.load(open(excluded_systems_tau_list_file_name, "rb"))
    import matplotlib.pyplot as plt
    excluded_systems_tau_list_per_system = [] # e.g. 0-> (10 (0.5), 20(0.75), 30(0.87), 40(0.9), 50 ,60)

    all_taus_for_system_numbers_across_all_shuffles = {} # key is system numbers # value is a list of numbers


    for excluded_systems_index, system_numbers_tau_list in excluded_systems_tau_list.iteritems():
        system_numbers_list = []
        means = []
        stds = []
        mins = []
        maxs = []
        for system_numbers, number_shuffle_tau_list in sorted(system_numbers_tau_list.iteritems()):
            system_numbers_list.append(system_numbers)
            print system_numbers, np.mean(number_shuffle_tau_list)
            if system_numbers in all_taus_for_system_numbers_across_all_shuffles:
                tau_list_tmp = all_taus_for_system_numbers_across_all_shuffles[system_numbers]
                for tau_value in number_shuffle_tau_list:
                    tau_list_tmp.append(tau_value)
                all_taus_for_system_numbers_across_all_shuffles[system_numbers] = tau_list_tmp
            else:
                tau_list_tmp = []
                for tau_value in number_shuffle_tau_list:
                    tau_list_tmp.append(tau_value)
                all_taus_for_system_numbers_across_all_shuffles[system_numbers] = tau_list_tmp

            means.append(np.mean(number_shuffle_tau_list))
            stds.append(np.std(number_shuffle_tau_list))
            mins.append(np.min(number_shuffle_tau_list))
            maxs.append(np.max(number_shuffle_tau_list))

        excluded_systems_tau_list_per_system.append(means)
        plt.errorbar(system_numbers_list, means, stds, fmt='ok', lw=3)
        plt.errorbar(system_numbers_list, means, [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
                     fmt='.k', ecolor='gray', lw=1)
        #plt.plot(system_numbers_list, means)

        plt.xlabel("number of systems")
        plt.ylabel("kendall's tau")
        plt.ylim([0.8,1])
        plt.legend()
        print (data_path)
        plt.savefig(data_path + 'tau_plots_new'+ str(excluded_systems_index) + '.png', bbox_inches='tight')
        plt.clf()

    y = np.array([np.array(list_item) for list_item in excluded_systems_tau_list_per_system])
    print y
    global_means = np.average(y, axis=0)  # average across 5 shuffles
    global_std = np.std(y, axis=0)


    plt.errorbar(np.arange(10,60,10), global_means, global_std, fmt='ok', lw=3)
    plt.xlabel("number of systems")
    plt.ylabel("kendall's tau")
    plt.ylim([0.8, 1])
    plt.legend()
    print (data_path)
    plt.savefig(data_path + 'tau_plots_global.png', bbox_inches='tight')
    plt.clf()


    import pandas as pd
    import collections
    od = collections.OrderedDict(sorted(all_taus_for_system_numbers_across_all_shuffles.items()))
    df = pd.DataFrame.from_dict(od, orient='index')
    print df

    system_numbers_list = list(sorted(all_taus_for_system_numbers_across_all_shuffles.iterkeys()))
    #for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
    plt.boxplot(df)
        # plt.xticks(x_labels, x_labels_set)
    plt.xlabel("number of systems")
    plt.ylabel("kendall's tau")
    plt.ylim([0.5, 1])
    plt.xticks(np.arange(1, len(system_numbers_list) + 1), system_numbers_list)
    plt.grid()
    # Save the figure

    all_taus_box_plot = "all_taus_box_plot.png"
    print data_path + all_taus_box_plot
    plt.savefig(data_path + all_taus_box_plot, bbox_inches='tight')
    # fig.clear()
    plt.clf()

    exit(0)
    '''
    for excluded_systems_index in excluded_systems_index_list:
        exlcuded_system_names = excluded_systems[excluded_systems_index]
        for number_shuffle in xrange(0,5):
            # we need to shuffle the system runs list 5 times
            # so e.g., for run 1--> system 1, 3 ,5 added
            # then for run 2--> system 3, 5, 1 added
            my_randoms = random.sample(xrange(len(system_runs_TREC8_list)), len(system_runs_TREC8_list))
            print my_randoms
            system_run_list_names = []
            for number in my_randoms:
                system_run_list_names.append(system_runs_TREC8_list[number])

            pseudo_qrels = {}  # topicID --> docNo, lable # will be updated each time a system is added
            j = 0 # for system number
            for i, system_name in enumerate(system_run_list_names):
                if system_name not in exlcuded_system_names:
                    # get the list of documents from that system within Rank 100
                    # topicId(str) --> (documentId(str) --> rank(int))
                    documentsFromSystem = systemRankedDocuments[system_name]
                    for topicNo, docNo_to_rank in sorted(documentsFromSystem.iteritems()):
                        for docNo in docNo_to_rank:
                            # if that docNo is in original_qrels construc qrels for that
                            if docNo in topic_qrels[topicNo]:
                                #print (system_name, docNo)
                                if topicNo in pseudo_qrels:
                                    docNo_label = pseudo_qrels[topicNo]
                                    docNo_label[docNo] = topic_qrels[topicNo][docNo]
                                    pseudo_qrels[topicNo] = docNo_label
                                else:
                                    docNo_label = {}
                                    docNo_label[docNo] = topic_qrels[topicNo][docNo]
                                    pseudo_qrels[topicNo] = docNo_label

                    total_len = 0
                    s = ""
                    for topicNo, docNo_label in sorted(pseudo_qrels.iteritems()):
                        total_len = total_len + len(pseudo_qrels[topicNo])
                        #print (len(pseudo_qrels[topicNo]))
                        for docNo, label in docNo_label.iteritems():
                            s = s + topicNo + " 0 "+ docNo + " "+ str(label) + "\n"
                    pseudo_qrels_file_name = data_path + datasource + "_pseudo_qrels_"+str(excluded_systems_index)+"_"+str(number_shuffle)+"_"+str(j)+".txt"
                    j = j + 1
                    f = open(pseudo_qrels_file_name, "w")
                    f.write(s)
                    f.close()

                    print(system_name, total_len)


    exit(0)
    '''
    all_info = {}
    global_tau_list = []
    global_drop_list = []
    global_delta_list = []

    '''
    #for excluded_systems_index in excluded_systems_index_list:
    for excluded_systems_index in [0]:

        #[0, 1, 3, 4]
        exlcuded_system_names = excluded_systems[excluded_systems_index]

        original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
            exlcuded_system_names, systemAddress[datasource], qrelAddress[datasource], rankMetric)
        original_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_original_' + rankMetric + '.pickle'
        pickle.dump(original_system_metric_value, open(original_system_metric_value_file_name, 'wb'))

        drop_list = []
        tau_list = []
        delta_score_list = []
        print ("================================")
        print ("exlcuded index", excluded_systems_index)


        for number_shuffle in xrange(0, 5):
            print ('shuffle number', number_shuffle)
            shuffule_tau_list = [] # for this shuffle we will have 51 tau values in a list
            shuffule_drop_list = []
            shuffle_delta_list = []
            for i in xrange(0, 51):
                relevanceJudgementAddress = data_path + datasource + "_pseudo_qrels_"+str(excluded_systems_index)+"_"+str(number_shuffle)+"_"+str(i)+".txt"
                print relevanceJudgementAddress
                predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
                    exlcuded_system_names, systemAddress[datasource], relevanceJudgementAddress, rankMetric)
                predicted_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_predicted_' + rankMetric + '_' + str(
                    i) + '.pickle'
                pickle.dump(predicted_system_metric_value, open(predicted_system_metric_value_file_name, 'wb'))
                tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)
                shuffule_tau_list.append(tau)
                drop_in_rank_list, delta_in_score_list = drop_calculator(original_system_metric_value_list,
                                                                         predicted_system_metric_value_list, i)
                shuffule_drop_list.append(drop_in_rank_list)
                shuffle_delta_list.append(delta_in_score_list)
                print excluded_systems_index, i, tau
            tau_list.append(shuffule_tau_list) # tau_list is a list of list [5][51]
            drop_list.append(shuffule_drop_list) # [5][51][20] # 4 excluded systesm, 51 system addeed and 20 system ranks
            delta_score_list.append(shuffle_delta_list)

        # 5 shuffle is finished now we have to take average across 5 shuffle
        # we can take average across tau only
        y = np.array([np.array(list_item) for list_item in tau_list])
        z = np.average(y, axis=0) # average across 5 shuffles

        global_tau_list.append(list(z))
        global_drop_list.append(drop_list)
        global_delta_list.append(delta_score_list)



        all_info[excluded_systems_index] = (global_tau_list, global_drop_list, global_delta_list)

        all_info_file_path = data_path + datasource + '_tau_drop_delta_list_' + str(excluded_systems_index) +'.pickle'
        pickle.dump(all_info, open(all_info_file_path, 'wb'))
    '''

    import matplotlib.pyplot as plt
    j = 0
    tau_values_from_systems = []
    for excluded_systems_index in excluded_systems_index_list:
        all_info_file_path = data_path + datasource + '_tau_drop_delta_list_' + str(excluded_systems_index) +'.pickle'
        all_info = pickle.load(open(all_info_file_path, 'rb'))
        tau_list = all_info[excluded_systems_index][0][0] # first zero for accessing tau_list, second zero is the only element of that list which is a list of 51 tau
        print (tau_list)
        tau_values_from_systems.append(tau_list)
        plt.plot(tau_list, label='Variation = ' + str(j))
        j = j + 1

    plt.xlabel("number of systems")
    plt.ylabel("kendall's tau")
    plt.legend()
    print (data_path)
    plt.savefig(data_path + 'tau_plots_shuffled_4_times.png', bbox_inches='tight')

    all_info_file_path = data_path + datasource + '_tau_drop_delta_list_' + str(0) +'.pickle'
    all_info = pickle.load(open(all_info_file_path, 'rb'))
    # drop_values_from_system is a list of drop_values at every point of 51 system added
    drop_values_from_system = all_info[0][1][0][0]  # 0the excluded system list(dictionary key), 1 for access global_drop_list, 0the exlcuded systems, 0the valriation of system
    #print len(drop_values_from_system), len(drop_values_from_system[0])

    plt.clf()

    # this is for getting tau_values when the total budget is equal to
    # pooled documents from systems (51 possible systems) not
    # in the excluded list [0] --> 20 systems specifically we took 1,
    # varied_pool_0 --> one system --> 9540 docs
    # varied_pool_1 --> 5 systems --> 16797 docs
    # varied_pool_2 --> 10 systems --> 22410 docs
    # varied_pool_3 --> 15 systems --> 26844 docs
    # varied_pool_4 --> 20 systems --> 30160 docs

    data_path = "/work/04549/mustaf/lonestar/data/TREC/TREC8/result/LR/varied_pool_"
    directory_list = [0,1,2,3,4]
    directory_index_systems_numbers = {}
    directory_index_systems_numbers[0] = 2
    directory_index_systems_numbers[1] = 5
    directory_index_systems_numbers[2] = 10
    directory_index_systems_numbers[3] = 15
    directory_index_systems_numbers[4] = 20

    poole_budget_in_directory = [7455, 13551, 24104, 32881, 43463]
    tau_values_from_al = []
    drop_values_from_al = []
    for directory_index in directory_list:
        varied_pooled_qrels_tau_path = data_path + str(directory_index) + "/IS_NR_CAL_tau_map.pickle"
        # suppose the budget is 9540 docs
        # taU-list contains list of tau_values at
        # 10, 20 ...100% of budget = 9540
        tau_list = pickle.load(open(varied_pooled_qrels_tau_path, 'rb'))
        plt.plot(tau_list, label='# of systems = ' + str(directory_index_systems_numbers[directory_index]) + ', pooled # of docs in qrel = ' + str(poole_budget_in_directory[directory_index]))
        print tau_list
        tau_values_from_al.append(tau_list[-1]) # getting the last elemene to tau_list whcih is at 100% of budgte

        # getting the drop_list alos
        varied_pooled_qrels_drop_path = data_path + str(directory_index) + "/IS_NR_CAL_droplist.pickle"
        drop_list = pickle.load(open(varied_pooled_qrels_drop_path, 'rb'))
        drop_values_from_al.append(drop_list[-1])


    print "tau from AL", tau_values_from_al
    plt.xlabel("percentage of human judgments + machine prediction")
    plt.ylabel("kendall's tau")
    plt.xticks(np.arange(len(x_labels_set)), x_labels_set)
    plt.ylim([0.5,1])
    plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,
            borderaxespad=0, frameon=False)
    #print (data_path)
    plt.savefig(data_path + 'tau_plots_from_AL.png', bbox_inches='tight')

    plt.clf()

    # because that is the worst performing systems list
    tau_values_from_sytems_index_0 = []
    drop_values_from_system_index_0 = []
    list_of_systems_used_AL = [0, 4, 9, 14, 19]
    for list_values in list_of_systems_used_AL:
        tau_values_from_sytems_index_0.append(tau_values_from_systems[0][list_values])
        drop_values_from_system_index_0.append(drop_values_from_system[list_values])

    print "tau_systems", tau_values_from_sytems_index_0
    plt.plot(tau_values_from_sytems_index_0, label='From pooled systems')
    plt.plot(tau_values_from_al, label='From AL system')
    plt.xlabel("number of systems")
    plt.ylabel("kendall's tau")
    plt.xticks(np.arange(5), [1,5,10,15,20])
    plt.ylim([0.5, 1])
    plt.legend()
    #print (data_path)
    plt.savefig(data_path + 'tau_plots_comparison_system_vs_al.png', bbox_inches='tight')

    plt.clf()

    print len(drop_values_from_al), len(drop_values_from_al[0])
    print len(drop_values_from_system_index_0), len(drop_values_from_system_index_0[0])

    data = {}
    data[2] = {}
    data[5] = {}
    data[10] = {}
    data[15] = {}
    data[20] = {}

    j = 0
    for k, v in data.iteritems():
        v['system'] = drop_values_from_system_index_0[j]
        v['AL'] = drop_values_from_al[j]
        j = j + 1

    fig, axes = plt.subplots(ncols=5, sharey=True)
    fig.subplots_adjust(wspace=0)

    for ax, name in zip(axes, [2,5,10,15,20]):
        ax.boxplot([data[name][item] for item in ['system', 'AL']])
        ax.set(xticklabels=['system', 'AL'], xlabel=str(name))
        ax.margins(0.05)

    #plt.ylim([0,15])
    plt.yticks(np.arange(15), np.arange(15))
    plt.savefig(data_path + 'drop_plots_comparison_system_vs_al.png', bbox_inches='tight')



    '''
    import matplotlib.pyplot as plt
    j = 0
    for k, v in all_info.iteritems():
        tau_list = v[0]
        print (tau_list)
        plt.plot(tau_list, label= 'Variation = ' + str(j))
        j = j + 1

    plt.xlabel("number of systems")
    plt.ylabel("kendall's tau")
    plt.legend()
    print (data_path)
    plt.savefig(data_path + 'tau_plots.png', bbox_inches='tight')


    plt.clf()

    x_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_labels_set = [10, 20, 30, 40, 50]

    j = 0
    for k, v in all_info.iteritems():
        drop_list = v[1]
        #print (drop_list)
        #plt.plot(drop_list)

        drop_fig_name = datasource + '_drop_rank_' + str(j) + '.png'
        j = j + 1
        # Create a figure instance
        #fig = plt.figure(1, figsize=(20, 20))
        # Create an axes instance
        #ax = fig.add_subplot(111)
        # Create the boxplot
        #bp = fig.boxplot(drop_list)
        plt.boxplot(drop_list)
        #plt.xticks(x_labels, x_labels_set)
        plt.xlabel("number of systems")
        plt.ylabel("drop in system ranking score")
        plt.ylim([0,20])
        plt.title("collection = " + datasource + "\n rank metric = " + rankMetric)

        # Save the figure
        plt.savefig(data_path + drop_fig_name, bbox_inches='tight')
        #fig.clear()
        plt.clf()

    '''
    '''
    classifier = None
    if classifier_name == 'LR':
        classifier = LogisticRegression(solver=small_data_solver,C=small_data_C_parameter)

    # loading metadata about the whole document collection
    # basically we are loading one list and one dictionary
    # list is a list of documentID from TREC which is docIndexToDocId
    # dictionary is a map from documentID to documentIndex, which is docIdToDocIndex
    metadata = pickle.load(open(source_file_path + meta_data_file_name[datasource], 'rb'))
    docIndexToDocId = metadata['docIndexToDocId']
    docIdToDocIndex = metadata['docIdToDocIndex']

    # creating a dictionary of key-->topicId, value-->original_labels
    # original_labels is a dictionary indexed by document_index from document_collection
    # and values is the doc_labels
    topic_original_qrels_in_doc_index = topicData.construct_original_qrels(docIdToDocIndex, topic_qrels, data_path, topic_original_qrels_in_doc_index_filename)

    documentIdListFromRanker = None
    if seed_selection_type == 'RDS':
        systemInfo = systemReader(datasource, start_topic[datasource], end_topic[datasource])
        documentIdListFromRanker = systemInfo.documentListFromSystem(systemAddress[datasource], systemName[datasource], data_path)

    # only IS and, RDS 
    topic_seed_info_file_name = "per_topic_seed_documents_" + seed_selection_type
    topic_seed_info = topicData.get_topic_seed_documents(topic_qrels, topic_original_qrels_in_doc_index, docIdToDocIndex, number_of_seeds,
                                                    seed_selection_type, data_path, topic_seed_info_file_name, documentIdListFromRanker)

    # loading whole document collection which is in a CSR format
    # took 12.3898720741
    start = time.time()
    print "loading document collection in: ", source_file_path + csr_matrix_file_name[datasource]
    document_collection = sp.load_npz(source_file_path + csr_matrix_file_name[datasource])
    print "Document collection loaded in:", (time.time() - start)

    topic_initial_info = topic_initial_task(topic_original_qrels_in_doc_index, topic_seed_info, document_collection)


    budget = 0
    print budget

    # to do
    # add an initializer using the topic_initia_info in a for loop with a dictinary keyed by topicId
    # for the train_index_list
    # so that we can use in the next
    # also write a function to calculate the accuracy of the per topic classifier.
    # that should two dictionaries 1) train_stats_acc, 2) test_stat_acc
    # should contains three entires : 1) f1, 2) precision, 3) recall


    # creating topic complete qrels
    # complete qrels = original qrels + predicted qrels
    # predicted qrels can be two types: i) prediction ii) treat all non-pooled document as non-relevant
    # took 17:51 MM:SS
    start = time.time()
    topic_complete_qrels = construct_predicted_qrels(classifier,classifier_name,document_collection,docIdToDocIndex, topic_qrels, data_path, topic_complete_qrels_filename)
    print "Finished topic_complete_qrels in:", (time.time() - start)

    topic_complete_qrels_address = data_path + "per_topic_complete_qrels_" + classifier_name +"_"

    start = time.time()
    #topic_list = [str(topicID) for topicID in xrange(start_topic[datasource], end_topic[datasource])]

    topic_list = [str(topicID) for topicID in xrange(start_top, end_top)]

    #topic_list = [str(topicID) for topicID in xrange(401, 402)]
    #topic_list = []
    #for topicID in xrange(434, 451):
    #    topic_list.append(str(topicID))

    topic_all_info_file_name = "per_topic_predictions_"+ seed_selection_type + "_" + classifier_name + "_" + al_protocol + "_"
    #print data_path + topic_all_info_file_name + topic_list[0] + '.pickle'
    topic_all_info = active_learning(topic_list, al_protocol, al_classifier, document_collection, topic_seed_info, topic_complete_qrels_address, train_per_centage, data_path, topic_all_info_file_name)

    # sanity check
    for topicId in topic_list:
        print "topicId:", topicId
        topic_all_info_file_name = data_path + "per_topic_predictions_" + seed_selection_type + "_" + classifier_name + "_" + al_protocol+"_"+ str(topicId) + ".pickle"
        topic_all_info = pickle.load(open(topic_all_info_file_name,'rb'))
        for k in sorted(topic_all_info.iterkeys()):
            print k, topic_all_info[k][0], topic_all_info[k][1], len(topic_all_info[k][4]), len(topic_all_info[k][5]), len(topic_all_info[k][6])
    '''
