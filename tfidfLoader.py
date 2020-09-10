from tqdm import tqdm
from gensim import corpora, similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.similarities import Similarity

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

# actual active learning for TREC is happening here for a particular topicID
# here we run for either all documents in the collection
# or all documents in the official qrels
def active_learning_multi_processing(topicId, al_protocol, al_classifier, document_collection, topic_seed_info, topic_complete_qrels_address, train_per_centage, use_pooled_budget, per_topic_budget_from_trec_qrels):
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
        # suppose original budget is 5,
        # then when train_index_list is 5, we cannot just turn off Active learning
        # we need to use that AL with train_index_list of size 5 to train use that to predict the rest
        # so we cannot exit at 5, we should exit at 5 + 1
        # that is the reason we set per_topic_budget_from_trec_qrels[topicId] + 1 where 1 is the batch_size
        # it means everything of pooled_budget size in the train_list so we need not tany training of the model
        # so break here
        if use_pooled_budget == 1 and per_topic_budget_from_trec_qrels[topicId] == len(train_index_list):
            break

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

        if use_pooled_budget == 1:
            #print "use pooled budget"
            size_limit = math.ceil(train_per_centage[loopCounter] * per_topic_budget_from_trec_qrels[topicId])
            print "size limit:", size_limit, "total_docs:", per_topic_budget_from_trec_qrels[topicId]

        else:
            size_limit = math.ceil(train_per_centage[loopCounter] * total_documents)
            print "size limit:", size_limit, "total_docs:", total_documents
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

def active_learning(topic_list, al_protocol, al_classifier, document_collection, topic_seed_info, topic_complete_qrels_address,train_per_centage, data_path, file_name, use_pooled_budget, per_topic_budget_from_trec_qrels):
    num_workers = None
    workers = ProcessPool(processes = 1)
    with tqdm(total=len(topic_list)) as pbar:
        partial_active_learning_multi_processing = partial(active_learning_multi_processing, al_protocol=al_protocol, al_classifier = al_classifier, document_collection=document_collection,topic_seed_info=topic_seed_info,topic_complete_qrels_address=topic_complete_qrels_address,train_per_centage=train_per_centage, use_pooled_budget=use_pooled_budget, per_topic_budget_from_trec_qrels=per_topic_budget_from_trec_qrels)
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


#########################################################################
# main script
# create all directories manually before running otherwise multi-processing will create lock condition
# for creating files

if __name__ == '__main__':
    datasource = sys.argv[1]  # can be 'TREC8','gov2', 'WT2013','WT2014'
    al_protocol = sys.argv[2]  # 'SAL', 'CAL', # SPL is not there yet
    seed_selection_type = sys.argv[3] # 'IS' only
    classifier_name = sys.argv[4] # "LR", "NR"--> means non-relevant all
    collection_size = sys.argv[5] # 'all', 'qrels' qrels --> means consider documents inseide qrels only
    al_classifier = sys.argv[6] # SVM, RF, NB and LR
    start_top = int(sys.argv[7])
    use_pooled_budget = int(sys.argv[8]) # 1 means use and 0 does not use that
    use_original_qrels = int(sys.argv[9]) # 1 means use original qrels, other value 0 means
    varied_qrels_directory_number = int(sys.argv[10]) # 1,2,3
    end_top = start_top + 1


    source_file_path =  base_address + datasource + "/"
    data_path = base_address + datasource + "/result/"
    if collection_size == 'qrels':
        source_file_path =  base_address + datasource + "/sparseTRECqrels/"
        data_path = base_address + datasource + "/sparseTRECqrels/" + "result/"

    data_path = data_path + al_classifier + "/"
    topic_budget_file_path =  base_address + datasource + "/"

    qrelAddress_path = None
    if use_original_qrels == 1:
        qrelAddress_path = qrelAddress[datasource]
    else:
        data_path = data_path + "varied_pool_" + str(varied_qrels_directory_number) + "/"
        qrelAddress_path = data_path + "relevance.txt"
        topic_budget_file_path = data_path

    print "qrel address path", qrelAddress_path
    print "source_file_path", source_file_path
    print "data_path", data_path
    print "budget_file", topic_budget_file_path + topic_budget_from_official_qrels_file_name



    topicData = TRECTopics(datasource, start_topic[datasource], end_topic[datasource])
    # topic_original_qrels_filename is a string defined in the global_definition.py file
    topic_qrels = topicData.qrelsReader(qrelAddress_path, data_path, topic_original_qrels_filename)
    #we need a separate function her becasue we want to read all documents in the whole
    # collection but we want to use a budget that is officially allocated per topic in TREC
    # topic_budget is a dictionarry where topicID --> (budget, # of relevant docs)
    per_topic_budget_from_trec_qrels = topicData.topicBudgetFromOfficialQrels(qrelAddress_path, topic_budget_file_path, topic_budget_from_official_qrels_file_name)

    for k, v in sorted(per_topic_budget_from_trec_qrels.iteritems()):
        print k, v

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


    '''
    # sanity check
    for topicId in sorted(topic_seed_info.iterkeys()):
       #print topicId, len(topic_seed_info[topicId]), topic_seed_info[topicId]
       print topicId, len(topic_seed_info[topicId])
    '''

    # loading whole document collection which is in a CSR format
    # took 12.3898720741
    start = time.time()
    print "loading document collection in: ", source_file_path + csr_matrix_file_name[datasource]
    document_collection = sp.load_npz(source_file_path + csr_matrix_file_name[datasource])
    print "Document collection loaded in:", (time.time() - start)

    topic_initial_info = topic_initial_task(topic_original_qrels_in_doc_index, topic_seed_info, document_collection)


    budget = 0
    '''sanity check
    for topicId in sorted(topic_initial_info.iterkeys()):
        per_topic_X, per_topic_y, per_topic_train_index_list, document_index_list, per_topic_seed_one_counter, per_topic_seed_zero_counter = topic_initial_info[topicId]
        print topicId, len(per_topic_train_index_list), per_topic_seed_one_counter, per_topic_seed_zero_counter
        budget = budget + len(per_topic_train_index_list)
    '''
    print budget

    # to do
    # add an initializer using the topic_initia_info in a for loop with a dictinary keyed by topicId
    # for the train_index_list
    # so that we can use in the next
    # also write a function to calculate the accuracy of the per topic classifier.
    # that should two dictionaries 1) train_stats_acc, 2) test_stat_acc
    # should contains three entires : 1) f1, 2) precision, 3) recall

    '''
    This is for active topic selection using only qrels
    start = time.time()
    batch_size = 1
    budget_increment = 500
    pick_classifier_method = 'max'
    topic_result_file_name = "per_topic_predictions_" + seed_selection_type + "_" + classifier_name + "_" + al_protocol + "_" + pick_classifier_method + "_batch_" + str(batch_size) + "_"
    active_topic_selection(datasource, topic_initial_info, batch_size, budget_increment, pick_classifier_method, data_path, topic_result_file_name)
    print "time required for Active topic selection: ", (time.time() - start)
    exit(0)
    '''


    # creating topic complete qrels
    # complete qrels = original qrels + predicted qrels
    # predicted qrels can be two types: i) prediction ii) treat all non-pooled document as non-relevant
    # took 17:51 MM:SS
    print "Generating per topic complete qrels"
    start = time.time()
    topic_complete_qrels = construct_predicted_qrels(classifier,classifier_name,document_collection,docIdToDocIndex, topic_qrels, data_path, topic_complete_qrels_filename)
    print "Finished topic_complete_qrels in:", (time.time() - start)

    topic_complete_qrels_address = data_path + "per_topic_complete_qrels_" + classifier_name +"_"

    start = time.time()
    #topic_list = [str(topicID) for topicID in xrange(start_topic[datasource], end_topic[datasource])]

    topic_list = [str(topicID) for topicID in xrange(start_top, end_top)]
    print "topic_list", topic_list

    #topic_list = [str(topicID) for topicID in xrange(401, 402)]
    #topic_list = []
    #for topicID in xrange(434, 451):
    #    topic_list.append(str(topicID))

    topic_all_info_file_name = "per_topic_predictions_"+ seed_selection_type + "_" + classifier_name + "_" + al_protocol + "_"
    #print data_path + topic_all_info_file_name + topic_list[0] + '.pickle'
    topic_all_info = active_learning(topic_list, al_protocol, al_classifier, document_collection, topic_seed_info, topic_complete_qrels_address, train_per_centage, data_path, topic_all_info_file_name, use_pooled_budget, per_topic_budget_from_trec_qrels)

    # sanity check
    for topicId in topic_list:
        print "topicId:", topicId
        topic_all_info_file_name = data_path + "per_topic_predictions_" + seed_selection_type + "_" + classifier_name + "_" + al_protocol+"_"+ str(topicId) + ".pickle"
        topic_all_info = pickle.load(open(topic_all_info_file_name,'rb'))
        for k in sorted(topic_all_info.iterkeys()):
            print k, topic_all_info[k][0], topic_all_info[k][1], len(topic_all_info[k][4]), len(topic_all_info[k][5]), len(topic_all_info[k][6])

