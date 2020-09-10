# import user pythons file
from topic_description import TRECTopics
from systemReader import systemReader
from global_definition import *
from qRelsProcessor import *
import random
from statistics import mean
import pickle
import os
import sys
import random
import numpy as np
from collections import OrderedDict
import copy

from sklearn.feature_extraction.text import TfidfVectorizer



def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude


def cosine_similarity_pairwise(sklearn_representation):
    skl_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(sklearn_representation):
        cos_sim_list = []
        for count_1, doc_1 in enumerate(sklearn_representation):
            # print cosine_similarity(doc_0, doc_1)
            # the lower the cosine similarity the diverse query is
            # so we are substracting from 1
            cos_sim_list.append(cosine_similarity(doc_0, doc_1))
        skl_tfidf_comparisons.append(mean(cos_sim_list))

    return skl_tfidf_comparisons

def calculate_query_diversity(topicPopularQueryInfoList, data_path, datasource, systemName):
    TopicDiversityInfoSorted = {}
    TopicDiversityInfoSorted_filename = data_path + datasource + "_" + systemName + "_" + "diverse_query_by_tfidfscore.pickle"
    TopicTFIDF_filename = data_path + datasource + "_" + systemName + "_" + "Topicqueriestfidfvector.pickle"

    TopicQueryTFIDF = {}
    print "topic_query_diversity_info_file:", TopicDiversityInfoSorted_filename
    if os.path.exists(TopicDiversityInfoSorted_filename):
        TopicDiversityInfoSorted = pickle.load(open(TopicDiversityInfoSorted_filename, "rb"))
        return TopicDiversityInfoSorted
    else:
        for topicNo, queryList in sorted(topicPopularQueryInfoList.iteritems()):
            # create the transform
            vectorizer = TfidfVectorizer(analyzer="word", \
                                             tokenizer=None, \
                                             preprocessor=None, \
                                             stop_words="english")

            # tokenize and build vocab
            bag_of_word = vectorizer.fit_transform(queryList)
            bag_of_word = bag_of_word.toarray()
            print "type:, bag_of_word", type(bag_of_word)
            TopicQueryTFIDF[topicNo] = copy.deepcopy(bag_of_word)

            skl_tfidf_comparisons = cosine_similarity_pairwise(bag_of_word)
            #print skl_tfidf_comparisons
            #print len(skl_tfidf_comparisons)
            #print topicNo, mean(skl_tfidf_comparisons[0:5]), mean(skl_tfidf_comparisons[-5:]), queryList[0:5] , queryList[-5:]
            queryToDiversityScore = {}
            for query_index, query in enumerate(queryList):
                queryToDiversityScore[query] = skl_tfidf_comparisons[query_index]

            queryToDiversityScoreSorted = OrderedDict(sorted(queryToDiversityScore.items(), key=lambda x: x[1]))

            for query, diversity_score in queryToDiversityScoreSorted.items():
                print query, diversity_score

            TopicDiversityInfoSorted[topicNo] = queryToDiversityScoreSorted

        pickle.dump(TopicDiversityInfoSorted, open(TopicDiversityInfoSorted_filename, "wb"))
        pickle.dump(TopicQueryTFIDF, open(TopicTFIDF_filename, "wb"))
        return TopicDiversityInfoSorted

import math
def rmse_calculation(original_qrels, pseudo_qrels):
    sum_of_squared_diff = 0.0
    number_of_topics = len(list(original_qrels.keys())) * 1.0

    total_relevant_count = 0
    pseudo_total_relevant_count = 0

    for topicNo, original_relevant_count in original_qrels.iteritems():
        if topicNo not in pseudo_qrels:
            continue
        pseudo_relevant_count = pseudo_qrels[topicNo]
        squared_diff =  (original_relevant_count - pseudo_relevant_count)*(original_relevant_count - pseudo_relevant_count)
        sum_of_squared_diff = sum_of_squared_diff + squared_diff

        total_relevant_count = total_relevant_count + original_relevant_count
        pseudo_total_relevant_count = pseudo_total_relevant_count + pseudo_relevant_count

    return math.sqrt(sum_of_squared_diff/number_of_topics), (pseudo_total_relevant_count*1.0)/total_relevant_count

def generate_diverse_query_variant_qrels(TopicDiversityInfoSorted, topicInfo, data_path, datasource, systemName, ordering_of_query):
    # generating query varinats using popular count
    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
        systemNameList[datasource], systemAddress[datasource], uqv_qrels_file_name, rankMetric)

    # generating query varinats using popular count
    topicQrelsVariants = {}  # is a Dictionary(topicId, DocList)
    tau_list = []
    drop_list = []
    rmse_list = []
    recall_list = []
    qrelsize_list = []

    for queryVariantsNumber in xrange(0, 50):
        for topicNo, queryToDiversityScore in sorted(TopicDiversityInfoSorted.iteritems()):
            # queryList is a list of queryVariant order by their popularity count
            # picking the query which index is equal queryVariantsNumber
            queryList = list(queryToDiversityScore.keys())
            '''
            # for sanity check
            print queryList
            for query, score in queryToDiversityScore.iteritems():
                print query, score
            '''
            if queryVariantsNumber >= len(queryList):
                continue
            if ordering_of_query == "most_diverse":
                queryVariantString = queryList[queryVariantsNumber]
            elif ordering_of_query == "least_diverse":
                queryVariantString = queryList[len(queryList) - 1 - queryVariantsNumber]
            #print ordering_of_query, queryVariantString
            # get the list of documents for this queryVariantString
            docList = topicInfo[topicNo][queryVariantString]

            queryInfo = {}
            if topicNo in topicQrelsVariants:
                queryInfo = topicQrelsVariants[topicNo]

            queryInfo[queryVariantString] = docList
            topicQrelsVariants[topicNo] = queryInfo
            #exit(0)
        # At this topicQrelsVariants contains documents for all topics from queryVariantNumber
        # so dump it in a qrels file using the TREC format

        s = ""
        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_" + str(
            queryVariantsNumber) + ".txt"
        print pseudo_qrels_file_name
        qrelsize = 0

        topic_to_RelevantDocCounts = {}
        pseudo_qrels_topic_relevant_counts_filename = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_" + str(
            queryVariantsNumber) + "_topic_relevantDocCounts.pickle"

        for topicNo, queryInfo in sorted(topicQrelsVariants.iteritems()):
            all_doc_list = []
            relevant_count = 0
            for queryVariant, docList in sorted(queryInfo.iteritems()):
                all_doc_list = all_doc_list + docList

            # finding uique doclist
            final_doc_list = list(set(all_doc_list))

            for docNo in final_doc_list:
                # getting labels from qrels
                # not all ranked document in qrels
                if docNo in topic_qrels[str(topicNo)]:
                    label = topic_qrels[str(topicNo)][docNo]
                    s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"
                    qrelsize = qrelsize + 1
                    relevant_count = relevant_count + label
            topic_to_RelevantDocCounts[topicNo] = relevant_count

        pickle.dump(topic_to_RelevantDocCounts, open(pseudo_qrels_topic_relevant_counts_filename, "wb"))

        f = open(pseudo_qrels_file_name, "w")
        f.write(s)
        f.close()

        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_" + str(
            queryVariantsNumber) + ".txt"

        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
            systemNameList[datasource], systemAddress[datasource], pseudo_qrels_file_name, rankMetric)

        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

        tau_list.append(tau)

        max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                              predicted_system_metric_value_list)

        drop_list.append(max_drop)
        rmse_val, recall_val = rmse_calculation(uqv_qrels_topic_relevantDocsCount, topic_to_RelevantDocCounts)
        rmse_list.append(rmse_val)
        recall_list.append(recall_val)
        qrelsize_list.append(qrelsize)
        print "variants:", queryVariantsNumber, "qrel_size:", qrelsize, "RMSE:", rmse_val, "tau:", tau, "drop:", max_drop, "recall:", recall_val

    all_info_list = (tau_list, drop_list, rmse_list, qrelsize_list, recall_list)
    all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_"+ ordering_of_query + "_all_info_list.pickle"

    pickle.dump(all_info_list, open(all_info_list_file_name, "wb"))


def generate_popular_query_variant_qrels(maxNumberOfVariantsAcrossAllTopics, topicPopularQueryInfoList, topicInfo, ordering_of_query):

    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
        systemNameList[datasource], systemAddress[datasource], uqv_qrels_file_name, rankMetric)

    # generating query varinats using popular count
    topicQrelsVariants = {}  # is a Dictionary(topicId, DocList)
    tau_list = []
    drop_list = []
    rmse_list = []
    recall_list = []
    qrelsize_list = []

    for queryVariantsNumber in xrange(0, maxNumberOfVariantsAcrossAllTopics):
        for topicNo, queryList in sorted(topicPopularQueryInfoList.iteritems()):
            # queryList is a list of queryVariant order by their popularity count
            # picking the query which index is equal queryVariantsNumber
            if queryVariantsNumber >= len(queryList):
                continue
            queryVariantString = None
            if ordering_of_query == "most_popular":
                queryVariantString = queryList[queryVariantsNumber]
            elif ordering_of_query == "least_popular":
                queryVariantString = queryList[len(queryList) - 1 - queryVariantsNumber]

            # get the list of documents for this queryVariantString
            docList = topicInfo[topicNo][queryVariantString]

            queryInfo = {}
            if topicNo in topicQrelsVariants:
                queryInfo = topicQrelsVariants[topicNo]

            queryInfo[queryVariantString] = docList
            topicQrelsVariants[topicNo] = queryInfo
        # At this topicQrelsVariants contains documents for all topics from queryVariantNumber
        # so dump it in a qrels file using the TREC format

        s = ""
        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query +"_" + str(
            queryVariantsNumber) + ".txt"
        print pseudo_qrels_file_name
        topic_to_RelevantDocCounts = {}
        pseudo_qrels_topic_relevant_counts_filename = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query +"_" + str(
            queryVariantsNumber) + "_topic_relevantDocCounts.pickle"

        qrelsize = 0
        for topicNo, queryInfo in sorted(topicQrelsVariants.iteritems()):
            all_doc_list = []
            relevant_count = 0

            for queryVariant, docList in sorted(queryInfo.iteritems()):
                all_doc_list = all_doc_list + docList

            # finding uique doclist
            final_doc_list = list(set(all_doc_list))

            for docNo in final_doc_list:
                # getting labels from qrels
                # not all ranked document in qrels
                if docNo in topic_qrels[str(topicNo)]:
                    label = topic_qrels[str(topicNo)][docNo]
                    s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"
                    qrelsize = qrelsize + 1
                    relevant_count = relevant_count + label
            topic_to_RelevantDocCounts[topicNo] = relevant_count

        pickle.dump(topic_to_RelevantDocCounts, open(pseudo_qrels_topic_relevant_counts_filename, "wb"))

        f = open(pseudo_qrels_file_name, "w")
        f.write(s)
        f.close()

        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_" + str(
            queryVariantsNumber) + ".txt"

        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
            systemNameList[datasource], systemAddress[datasource], pseudo_qrels_file_name, rankMetric)

        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

        tau_list.append(tau)

        max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                              predicted_system_metric_value_list)

        drop_list.append(max_drop)
        qrelsize_list.append(qrelsize)

        rmse_val, recall_val = rmse_calculation(uqv_qrels_topic_relevantDocsCount, topic_to_RelevantDocCounts)
        rmse_list.append(rmse_val)
        recall_list.append(recall_val)
        print "variants:", queryVariantsNumber, "qrel_size:", qrelsize, "RMSE:", rmse_val, "tau:", tau, "drop:", max_drop, "recall:", recall_val

    all_info_list = (tau_list, drop_list, rmse_list, qrelsize_list, recall_list)
    all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_all_info_list.pickle"

    pickle.dump(all_info_list, open(all_info_list_file_name, "wb"))



def generate_soumya_query_variant_qrels(maxNumberOfVariantsAcrossAllTopics, x, topicPopularQueryInfoList, topicInfo, ordering_of_query):

    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
        systemNameList[datasource], systemAddress[datasource], uqv_qrels_file_name, rankMetric)

    # generating query varinats using popular count
    topicQrelsVariants = {}  # is a Dictionary(topicId, DocList)
    tau_list = []
    drop_list = []
    rmse_list = []
    recall_list = []
    qrelsize_list = []

    for queryVariantsNumber in xrange(0, maxNumberOfVariantsAcrossAllTopics):
        for topicNo, queryIndexList in sorted(x.iteritems()):
            queryList = topicPopularQueryInfoList[topicNo]
            #print topicNo, len(queryIndexList), len(queryList)

            # queryList is a list of queryVariant order by their popularity count
            # picking the query which index is equal queryVariantsNumber
            if queryVariantsNumber >= len(queryList):
                continue
            queryVariantString = None
            if ordering_of_query == "most_popular":
                #print queryIndexList[queryVariantsNumber], len(queryList)
                queryVariantString = queryList[queryIndexList[queryVariantsNumber] - 1] # because soumya did 1 indexing
            elif ordering_of_query == "least_popular":
                queryVariantString = queryList[len(queryList) - 1 - queryVariantsNumber]

            # get the list of documents for this queryVariantString
            docList = []

            if queryVariantString in topicInfo[topicNo]:
                docList = topicInfo[topicNo][queryVariantString]
            else:
                print topicNo, queryVariantString
            queryInfo = {}
            if topicNo in topicQrelsVariants:
                queryInfo = topicQrelsVariants[topicNo]

            queryInfo[queryVariantString] = docList
            topicQrelsVariants[topicNo] = queryInfo
        # At this topicQrelsVariants contains documents for all topics from queryVariantNumber
        # so dump it in a qrels file using the TREC format

        s = ""
        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_soumya_" + ordering_of_query +"_" + str(
            queryVariantsNumber) + ".txt"
        print pseudo_qrels_file_name
        topic_to_RelevantDocCounts = {}
        pseudo_qrels_topic_relevant_counts_filename = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_soumya_" + ordering_of_query +"_" + str(
            queryVariantsNumber) + "_topic_relevantDocCounts.pickle"

        qrelsize = 0
        for topicNo, queryInfo in sorted(topicQrelsVariants.iteritems()):
            all_doc_list = []
            relevant_count = 0

            for queryVariant, docList in sorted(queryInfo.iteritems()):
                all_doc_list = all_doc_list + docList

            # finding uique doclist
            final_doc_list = list(set(all_doc_list))

            for docNo in final_doc_list:
                # getting labels from qrels
                # not all ranked document in qrels
                if docNo in topic_qrels[str(topicNo)]:
                    label = topic_qrels[str(topicNo)][docNo]
                    s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"
                    qrelsize = qrelsize + 1
                    relevant_count = relevant_count + label
            topic_to_RelevantDocCounts[topicNo] = relevant_count

        pickle.dump(topic_to_RelevantDocCounts, open(pseudo_qrels_topic_relevant_counts_filename, "wb"))

        f = open(pseudo_qrels_file_name, "w")
        f.write(s)
        f.close()

        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_soumya_" + ordering_of_query + "_" + str(
            queryVariantsNumber) + ".txt"

        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
            systemNameList[datasource], systemAddress[datasource], pseudo_qrels_file_name, rankMetric)

        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

        tau_list.append(tau)

        max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                              predicted_system_metric_value_list)

        drop_list.append(max_drop)
        qrelsize_list.append(qrelsize)

        rmse_val, recall_val = rmse_calculation(uqv_qrels_topic_relevantDocsCount, topic_to_RelevantDocCounts)
        rmse_list.append(rmse_val)
        recall_list.append(recall_val)
        print "variants:", queryVariantsNumber, "qrel_size:", qrelsize, "RMSE:", rmse_val, "tau:", tau, "drop:", max_drop, "recall:", recall_val

    all_info_list = (tau_list, drop_list, rmse_list, qrelsize_list, recall_list)
    all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_soumya_" + ordering_of_query + "_all_info_list.pickle"

    pickle.dump(all_info_list, open(all_info_list_file_name, "wb"))



def generate_alex_query_variant_qrels(maxNumberOfVariantsAcrossAllTopics, topic_set_queryList, topicPopularQueryInfoList, topicInfo, ordering_of_query):

    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
        systemNameList[datasource], systemAddress[datasource], uqv_qrels_file_name, rankMetric)

    # generating query varinats using popular count
    tau_list = []
    drop_list = []
    rmse_list = []
    recall_list = []
    qrelsize_list = []

    for queryVariantsNumber in xrange(0, maxNumberOfVariantsAcrossAllTopics):
        topicQrelsVariants = {}  # is a Dictionary(topicId, DocList)

        for topicNo, set_to_queryList in sorted(topic_set_queryList.iteritems()):

            queryList = []
            if queryVariantsNumber + 1 > len(set_to_queryList):
                queryList = set_to_queryList[len(set_to_queryList)] # taking last
            else:
                queryList = set_to_queryList[queryVariantsNumber + 1] # since Alex set starts from 1
            #print topicNo, len(queryIndexList), len(queryList)

            for queryVariantString in queryList:
                # get the list of documents for this queryVariantString
                docList = []

                if queryVariantString in topicInfo[topicNo]:
                    docList = topicInfo[topicNo][queryVariantString]
                else:
                    print topicNo, queryVariantString
                queryInfo = {}
                if topicNo in topicQrelsVariants:
                    queryInfo = topicQrelsVariants[topicNo]

                queryInfo[queryVariantString] = docList
                topicQrelsVariants[topicNo] = queryInfo
            # At this topicQrelsVariants contains documents for all topics from queryVariantNumber
        # so dump it in a qrels file using the TREC format

        s = ""
        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_alex_" + ordering_of_query +"_" + str(
            queryVariantsNumber) + ".txt"
        print pseudo_qrels_file_name
        topic_to_RelevantDocCounts = {}
        pseudo_qrels_topic_relevant_counts_filename = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_alex_" + ordering_of_query +"_" + str(
            queryVariantsNumber) + "_topic_relevantDocCounts.pickle"

        qrelsize = 0
        for topicNo, queryInfo in sorted(topicQrelsVariants.iteritems()):
            all_doc_list = []
            relevant_count = 0

            for queryVariant, docList in sorted(queryInfo.iteritems()):
                all_doc_list = all_doc_list + docList

            # finding uique doclist
            final_doc_list = list(set(all_doc_list))

            for docNo in final_doc_list:
                # getting labels from qrels
                # not all ranked document in qrels
                if docNo in topic_qrels[str(topicNo)]:
                    label = topic_qrels[str(topicNo)][docNo]
                    s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"
                    qrelsize = qrelsize + 1
                    relevant_count = relevant_count + label
            topic_to_RelevantDocCounts[topicNo] = relevant_count

        pickle.dump(topic_to_RelevantDocCounts, open(pseudo_qrels_topic_relevant_counts_filename, "wb"))

        f = open(pseudo_qrels_file_name, "w")
        f.write(s)
        f.close()

        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_alex_" + ordering_of_query + "_" + str(
            queryVariantsNumber) + ".txt"

        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
            systemNameList[datasource], systemAddress[datasource], pseudo_qrels_file_name, rankMetric)

        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

        tau_list.append(tau)

        max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                              predicted_system_metric_value_list)

        drop_list.append(max_drop)
        qrelsize_list.append(qrelsize)

        rmse_val, recall_val = rmse_calculation(uqv_qrels_topic_relevantDocsCount, topic_to_RelevantDocCounts)
        rmse_list.append(rmse_val)
        recall_list.append(recall_val)
        print "variants:", queryVariantsNumber, "qrel_size:", qrelsize, "RMSE:", rmse_val, "tau:", tau, "drop:", max_drop, "recall:", recall_val

    all_info_list = (tau_list, drop_list, rmse_list, qrelsize_list, recall_list)
    all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_alex_" + ordering_of_query + "_all_info_list.pickle"

    pickle.dump(all_info_list, open(all_info_list_file_name, "wb"))




def qrel_file_generator_all_HIL(budget_topic_docs, document_selection_type, data_path, datasource, systemName):

    list_of_budget = list(sorted(budget_topic_docs.keys()))
    list_of_topics = list(sorted(budget_topic_docs[list_of_budget[0]].keys()))
    #list_of_pool_depth = list(sorted(topicPoolDocuments[list_of_topics[0]].keys()))

    tau_list = []
    drop_list = []
    relevant_count_list = []
    qrelsize_list = []
    recall_ratio_list = []

    ## generating the uqv_qrels file using the last budget point
    ## because that is pool_depth we have used for all document selection

    s = ""
    last_budget = list_of_budget[-1]
    uqv_qrels_pool_10_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "_" + str(
        last_budget) + ".txt"

    total_relevant_docs = 0
    for topicNo in sorted(list_of_topics):
        # print pool_depth, topicNo
        list_of_documents = budget_topic_docs[last_budget][topicNo]

        for docNo in list_of_documents:
            # getting labels from qrels
            # not all ranked document in qrels
            if docNo in topic_qrels[str(topicNo)]:
                label = topic_qrels[str(topicNo)][docNo]
                total_relevant_docs = total_relevant_docs + label
                s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"


    f = open(uqv_qrels_pool_10_file_name, "w")
    f.write(s)
    f.close()
    total_relevant_docs = 1.0*total_relevant_docs

    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
        systemNameList[datasource], systemAddress[datasource], uqv_qrels_pool_10_file_name, rankMetric)



    for current_budget in list_of_budget:
        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" +document_selection_type + "_" + str(current_budget) + ".txt"

        s = ""
        relevant_count = 0
        qrelsize = 0

        for topicNo in sorted(list_of_topics):
            # print pool_depth, topicNo
            list_of_documents = budget_topic_docs[current_budget][topicNo]

            for docNo in list_of_documents:
                # getting labels from qrels
                # not all ranked document in qrels
                if docNo in topic_qrels[str(topicNo)]:
                    label = topic_qrels[str(topicNo)][docNo]
                    relevant_count = relevant_count + label
                    s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"
                    qrelsize = qrelsize + 1

        f = open(pseudo_qrels_file_name, "w")
        f.write(s)
        f.close()

        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "_" + str(
            current_budget) + ".txt"

        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
            systemNameList[datasource], systemAddress[datasource], pseudo_qrels_file_name, rankMetric)

        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

        tau_list.append(tau)

        max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                              predicted_system_metric_value_list)

        drop_list.append(max_drop)
        qrelsize_list.append(qrelsize)
        relevant_count_list.append(relevant_count)
        recall_val = (relevant_count*1.0)/total_relevant_docs
        recall_ratio_list.append(recall_val)

        #rmse_val = rmse_calculation(uqv_qrels_topic_relevantDocsCount, topic_to_RelevantDocCounts)
        #rmse_list.append(rmse_val)
        print "current_budget:", current_budget, "qrel_size:", qrelsize, "releCount", relevant_count, "tau:", tau, "drop:", max_drop, "recall:", recall_val

    all_info_list = (tau_list, drop_list, relevant_count_list, recall_ratio_list, qrelsize_list, list_of_budget)
    all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "all_info_list.pickle"
    print all_info_list_file_name
    pickle.dump(all_info_list, open(all_info_list_file_name, "wb"))



def qrel_file_generator_all(topicPoolDocuments, document_selection_type, data_path, datasource, systemName):
    list_of_topics = list(sorted(topicPoolDocuments.keys()))
    list_of_pool_depth = list(sorted(topicPoolDocuments[list_of_topics[0]].keys()))

    tau_list = []
    drop_list = []
    relevant_count_list = []
    qrelsize_list = []
    recall_ratio_list = []

    ## generating the uqv_qrels file using pool_depth 10
    ## because that is pool_depth we have used for all document selection

    s = ""
    pool_depth = 10
    uqv_qrels_pool_10_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "_" + str(
        pool_depth) + ".txt"

    total_relevant_docs = 0
    for topicNo in list_of_topics:
        # print pool_depth, topicNo
        list_of_documents = topicPoolDocuments[topicNo][pool_depth-1][0]
        total_relevant_docs = total_relevant_docs + topicPoolDocuments[topicNo][pool_depth-1][1]

        for docNo in list_of_documents:
            # getting labels from qrels
            # not all ranked document in qrels
            if docNo in topic_qrels[str(topicNo)]:
                label = topic_qrels[str(topicNo)][docNo]
                s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"

    f = open(uqv_qrels_pool_10_file_name, "w")
    f.write(s)
    f.close()
    total_relevant_docs = 1.0*total_relevant_docs

    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
        systemNameList[datasource], systemAddress[datasource], uqv_qrels_pool_10_file_name, rankMetric)

    qrelsize = 0

    for pool_depth in list_of_pool_depth:
        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" +document_selection_type + "_" + str(pool_depth) + ".txt"

        s = ""
        relevant_count = 0

        for topicNo in list_of_topics:
            #print pool_depth, topicNo
            list_of_documents = topicPoolDocuments[topicNo][pool_depth][0]
            relevant_count = relevant_count + topicPoolDocuments[topicNo][pool_depth][1]
            for docNo in list_of_documents:
                # getting labels from qrels
                # not all ranked document in qrels
                if docNo in topic_qrels[str(topicNo)]:
                    label = topic_qrels[str(topicNo)][docNo]
                    s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"
                    qrelsize = qrelsize + 1

        f = open(pseudo_qrels_file_name, "w")
        f.write(s)
        f.close()

        pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "_" + str(
            pool_depth) + ".txt"

        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
            systemNameList[datasource], systemAddress[datasource], pseudo_qrels_file_name, rankMetric)

        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

        tau_list.append(tau)

        max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                              predicted_system_metric_value_list)

        drop_list.append(max_drop)
        qrelsize_list.append(qrelsize)
        relevant_count_list.append(relevant_count)
        recall_val = (relevant_count*1.0)/total_relevant_docs
        recall_ratio_list.append(recall_val)

        #rmse_val = rmse_calculation(uqv_qrels_topic_relevantDocsCount, topic_to_RelevantDocCounts)
        #rmse_list.append(rmse_val)
        print "pool_depth:", pool_depth, "qrel_size:", qrelsize, "releCount", relevant_count, "tau:", tau, "drop:", max_drop, "recall:", recall_val

    all_info_list = (tau_list, drop_list, relevant_count_list, recall_ratio_list, qrelsize_list)
    all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "all_info_list.pickle"
    print all_info_list_file_name
    pickle.dump(all_info_list, open(all_info_list_file_name, "wb"))


# it should call only once because it will be changed based on the random numbers
def random_query_variants_qrels(topicPopularQueryInfoList, topicInfo, data_path, datasource, systemName):

    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
        systemNameList[datasource], systemAddress[datasource], uqv_qrels_file_name, rankMetric)

    numberOfSamples = 1000

    topicQrelsVariants = {}  # is a Dictionary(topicId, DocList)


    random.seed(30)
    samples_to_values = {} # key is the sample number and values are tau_list, drop_list, etc
    for sample_number in xrange(1, numberOfSamples + 1):
        # iterating over number of sample
        uniqueRelevantDocList = []

        tau_list = [] # list of taus for each query variants
        drop_list = []
        rmse_list = []
        recall_list = []
        qrelsize_list = []

        for queryVariantsNumber in xrange(0, 50):

            # clearning topicQrelsVariants every time
            topicQrelsVariants = {}

            for topicNo, queryList in sorted(topicPopularQueryInfoList.iteritems()):
                # queryList is a list of queryVariant order by their popularity count
                # picking the query which index is equal queryVariantsNumber
                queryVariantsNumberToUse = queryVariantsNumber + 1
                if queryVariantsNumber >= len(queryList):
                    queryVariantsNumberToUse = len(queryList)

                random_query_index_list = random.sample(xrange(len(queryList)), queryVariantsNumberToUse)
                # for each query variants we are putting the values
                for random_query_index in random_query_index_list:
                    queryVariantString = queryList[random_query_index]
                    # get the list of documents for this queryVariantString
                    docList = topicInfo[topicNo][queryVariantString]

                    queryInfo = {}
                    if topicNo in topicQrelsVariants:
                        queryInfo = topicQrelsVariants[topicNo]

                    queryInfo[queryVariantString] = docList
                    topicQrelsVariants[topicNo] = queryInfo
            # At this topicQrelsVariants contains documents for all topics from queryVariantNumber
            # so dump it in a qrels file using the TREC format
            s = ""
            pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_" + str(
                queryVariantsNumber) + "_sample_number_" + str(sample_number) + ".txt"
            print pseudo_qrels_file_name
            qrelsize = 0
            topic_to_RelevantDocCounts = {}
            for topicNo, queryInfo in sorted(topicQrelsVariants.iteritems()):
                all_doc_list = []
                uniqueRelevantDocCount = 0

                for queryVariant, docList in sorted(queryInfo.iteritems()):
                    all_doc_list = all_doc_list + docList

                # finding uique doclist
                final_doc_list = list(set(all_doc_list))

                for docNo in final_doc_list:
                    # getting labels from qrels
                    # not all ranked document in qrels
                    if docNo in topic_qrels[str(topicNo)]:
                        label = topic_qrels[str(topicNo)][docNo]
                        s = s + str(topicNo) + " 0 " + docNo + " " + str(label) + "\n"
                        qrelsize = qrelsize + 1
                        if label == 1:
                            uniqueRelevantDocCount = uniqueRelevantDocCount + 1
                topic_to_RelevantDocCounts[topicNo] = uniqueRelevantDocCount

            f = open(pseudo_qrels_file_name, "w")
            f.write(s)
            f.close()
            uniqueRelevantDocList.append(uniqueRelevantDocCount)

            #print "number of query variants:", queryVariantsNumber, "sample_number", sample_number, "qrel_size:", qrelsize, "unique rele count", uniqueRelevantDocCount  # , "tau:", tau

            predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
                systemNameList[datasource], systemAddress[datasource], pseudo_qrels_file_name, rankMetric)

            tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

            tau_list.append(tau)

            max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                                  predicted_system_metric_value_list)

            drop_list.append(max_drop)
            qrelsize_list.append(qrelsize)

            rmse_val, recall_val = rmse_calculation(uqv_qrels_topic_relevantDocsCount, topic_to_RelevantDocCounts)
            rmse_list.append(rmse_val)
            recall_list.append(recall_val)
            print "sample number:", sample_number, "variants:", queryVariantsNumber, "qrel_size:", qrelsize, "RMSE:", rmse_val, "tau:", tau, "drop:", max_drop, "recall:", recall_val


        all_info_list = (tau_list, drop_list, rmse_list, qrelsize_list, recall_list)
        all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_random_sample_" + str(sample_number) + "_all_info_list.pickle"

        pickle.dump(all_info_list, open(all_info_list_file_name, "wb"))


# topicInfo --> per topic Ranked List of Documents for each query variants
# topic_qrels original qrels file
# topicPoolInf --> pool wise budget info
def query_variants_bandit(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value):
    np.random.seed(3)

    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    #queryList = topicSortedInfo[topicId]

    poole_budget = topicPoolInfo[topicId]
    max_number_of_samples = pool_value
    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = [] # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = [] # key is the index of query same as queryList and value is how many times it has been sampled so far
    posterior_distribution = []
    for query_index in xrange(0, len(queryList)):
        relevant_count = 1
        nonrelevant_count = 1
        query_count.append([relevant_count, nonrelevant_count])
        query_sampled.append(0) # initially no sampled at all
        posterior_distribution.append(np.random.beta(query_count[query_index][0], query_count[query_index][1]))

    previous_pool_budget = 0
    # when pool_depth = 1, pool_docList will conatains all dcoument from pool_1
    # when pool_depth = 2, pool_docList will conatains all dcoument from pool_1 and pool_2
    # when pool_depth = n, pool_docList will conatains all dcoument from pool_1,pool_2, ..., pool_n


    pooled_docList = []
    pooled_Document = {} # key pool and values --> (pooled_docList, relevant_doc_count in that pool_docList)
    relevant_doc_count = 0

    for pool, infoList in sorted(poole_budget.iteritems()):
        total_budget = infoList[0] # per pooled budget
        budget_tracker = previous_pool_budget
        #print "pool:", pool, "total budget", total_budget
        while budget_tracker < total_budget:
            sampled_query_index = np.argmax(posterior_distribution)

            # if any of the query hits the pool_value which is 10 here
            # set its posterior distribution to 0.0
            # so that we will not sample from that query variants
            if query_sampled[sampled_query_index] == pool_value:
                posterior_distribution[sampled_query_index] = 0.0
                continue
            # get the next available documents for the query
            sampled_query_string = queryList[sampled_query_index]
            #print sampled_query_string,  query_sampled[sampled_query_index]
            sampled_document = topicInfo[topicId][sampled_query_string][query_sampled[sampled_query_index]]
            #print sampled_query_string, query_sampled[sampled_query_index], sampled_document

            if sampled_document not in pooled_docList:
                pooled_docList.append(sampled_document)
                docLabel = topic_qrels[str(topicId)][sampled_document]
                relevant_doc_count = relevant_doc_count + docLabel
                # only update the budget if it is a new document
                budget_tracker = budget_tracker + 1

            # if document is already in docList or not
            # the queryvariants should get credit always
            # increase the pointer of document for the query to the next
            query_sampled[sampled_query_index] = query_sampled[sampled_query_index] + 1
            # get the label
            docLabel = topic_qrels[str(topicId)][sampled_document]
            # update the query_count relevant and non-relevant count
            query_count[sampled_query_index] = [query_count[sampled_query_index][0] + docLabel,
                                                query_count[sampled_query_index][1] + docLabel]

            #update the posterior distribution
            posterior_distribution = []
            for query_index in xrange(0, len(queryList)):
                #np.random.seed(0)
                posterior_distribution.append(np.random.beta(query_count[query_index][0], query_count[query_index][1]))


        previous_pool_budget = budget_tracker

        this_level_pooled_docs_list = copy.deepcopy(pooled_docList)
        pooled_Document[pool] = [this_level_pooled_docs_list, relevant_doc_count]

    return pooled_Document

    for pool, document_list in sorted(pooled_Document.iteritems()):
        print pool, len(document_list[0]), document_list[1], topicPoolInfo[topicId][pool][0], topicPoolInfo[topicId][pool][1]



# topicInfo --> per topic Ranked List of Documents for each query variants
# topic_qrels original qrels file
# topicPoolInf --> pool wise budget info
def query_variants_bandit_ns(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    #queryList = topicSortedInfo[topicId]

    poole_budget = topicPoolInfo[topicId]
    max_number_of_samples = pool_value
    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = [] # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = [] # key is the index of query same as queryList and value is how many times it has been sampled so far
    query_retrieved_judged = []
    posterior_distribution = []
    for query_index in xrange(0, len(queryList)):
        relevant_count = 1
        nonrelevant_count = 1
        query_count.append([relevant_count, nonrelevant_count])
        query_sampled.append(0) # initially no sampled at all
        np.random.seed(3)
        query_retrieved_judged.append(2) # one relevant and one non relevant
        posterior_distribution.append(np.random.beta(query_count[query_index][0], query_retrieved_judged[query_index] - query_count[query_index][0]))

    previous_pool_budget = 0
    # when pool_depth = 1, pool_docList will conatains all dcoument from pool_1
    # when pool_depth = 2, pool_docList will conatains all dcoument from pool_1 and pool_2
    # when pool_depth = n, pool_docList will conatains all dcoument from pool_1,pool_2, ..., pool_n


    pooled_docList = []
    pooled_Document = {} # key pool and values --> (pooled_docList, relevant_doc_count in that pool_docList)
    relevant_doc_count = 0

    for pool, infoList in sorted(poole_budget.iteritems()):
        total_budget = infoList[0] # per pooled budget
        budget_tracker = previous_pool_budget
        #print "pool:", pool, "total budget", total_budget
        while budget_tracker < total_budget:
            sampled_query_index = np.argmax(posterior_distribution)

            # if any of the query hits the pool_value which is 10 here
            # set its posterior distribution to 0.0
            # so that we will not sample from that query variants
            if query_sampled[sampled_query_index] == pool_value:
                posterior_distribution[sampled_query_index] = 0.0
                continue
            # get the next available documents for the query
            sampled_query_string = queryList[sampled_query_index]
            #print sampled_query_string,  query_sampled[sampled_query_index]
            sampled_document = topicInfo[topicId][sampled_query_string][query_sampled[sampled_query_index]]
            if sampled_document not in pooled_docList:
                pooled_docList.append(sampled_document)
                docLabel = topic_qrels[str(topicId)][sampled_document]
                relevant_doc_count = relevant_doc_count + docLabel
                # only update the budget if it is a new document
                budget_tracker = budget_tracker + 1

            # if document is already in docList or not
            # the queryvariants should get credit always
            # increase the pointer of document for the query to the next
            query_sampled[sampled_query_index] = query_sampled[sampled_query_index] + 1
            # get the label
            docLabel = topic_qrels[str(topicId)][sampled_document]
            # update the query_count relevant and non-relevant count

            query_count[sampled_query_index] = [query_count[sampled_query_index][0] + docLabel,
                                                query_count[sampled_query_index][1] + docLabel]
            query_retrieved_judged[sampled_query_index] = query_retrieved_judged[sampled_query_index] + 1
            #update the posterior distribution
            posterior_distribution = []
            for query_index in xrange(0, len(queryList)):
                #np.random.seed(0)
                posterior_distribution.append(np.random.beta(query_count[query_index][0], query_retrieved_judged[query_index] - query_count[query_index][0]))


        previous_pool_budget = budget_tracker
        this_level_pooled_docs_list = copy.deepcopy(pooled_docList)
        pooled_Document[pool] = [this_level_pooled_docs_list, relevant_doc_count]

    return pooled_Document


    for pool, document_list in sorted(pooled_Document.iteritems()):
        print pool, len(document_list[0]), document_list[1], topicPoolInfo[topicId][pool][0], topicPoolInfo[topicId][pool][1]






# topicInfo --> per topic Ranked List of Documents for each query variants
# topic_qrels original qrels file
# topicPoolInf --> pool wise budget info
def query_variants_bandit_ns_v1(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    #queryList = topicSortedInfo[topicId]

    poole_budget = topicPoolInfo[topicId]
    max_number_of_samples = pool_value
    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = [] # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = [] # key is the index of query same as queryList and value is how many times it has been sampled so far
    query_retrieved_judged = []
    posterior_distribution = []
    query_docLabel = []
    for query_index in xrange(0, len(queryList)):
        relevant_count = 1
        nonrelevant_count = 1
        query_count.append([relevant_count, nonrelevant_count])
        query_sampled.append(0) # initially no sampled at all
        query_docLabel.append(1)
        np.random.seed(3)
        query_retrieved_judged.append(2) # one relevant and one non relevant
        posterior_distribution.append(np.random.beta(query_docLabel[query_index], 1))

    previous_pool_budget = 0
    # when pool_depth = 1, pool_docList will conatains all dcoument from pool_1
    # when pool_depth = 2, pool_docList will conatains all dcoument from pool_1 and pool_2
    # when pool_depth = n, pool_docList will conatains all dcoument from pool_1,pool_2, ..., pool_n


    pooled_docList = []
    pooled_Document = {} # key pool and values --> (pooled_docList, relevant_doc_count in that pool_docList)
    relevant_doc_count = 0

    for pool, infoList in sorted(poole_budget.iteritems()):
        total_budget = infoList[0] # per pooled budget
        budget_tracker = previous_pool_budget
        #print "pool:", pool, "total budget", total_budget
        while budget_tracker < total_budget:
            sampled_query_index = np.argmax(posterior_distribution)

            # if any of the query hits the pool_value which is 10 here
            # set its posterior distribution to 0.0
            # so that we will not sample from that query variants
            if query_sampled[sampled_query_index] == pool_value:
                posterior_distribution[sampled_query_index] = 0.0
                continue
            # get the next available documents for the query
            sampled_query_string = queryList[sampled_query_index]
            #print sampled_query_string,  query_sampled[sampled_query_index]
            sampled_document = topicInfo[topicId][sampled_query_string][query_sampled[sampled_query_index]]
            if sampled_document not in pooled_docList:
                pooled_docList.append(sampled_document)
                docLabel = topic_qrels[str(topicId)][sampled_document]
                relevant_doc_count = relevant_doc_count + docLabel
                # only update the budget if it is a new document
                budget_tracker = budget_tracker + 1

            # if document is already in docList or not
            # the queryvariants should get credit always
            # increase the pointer of document for the query to the next
            query_sampled[sampled_query_index] = query_sampled[sampled_query_index] + 1
            # get the label
            docLabel = topic_qrels[str(topicId)][sampled_document]
            # update the query_count relevant and non-relevant count
            query_docLabel[sampled_query_index] = docLabel
            query_count[sampled_query_index] = [query_count[sampled_query_index][0] + docLabel,
                                                query_count[sampled_query_index][1] + docLabel]
            query_retrieved_judged[sampled_query_index] = query_retrieved_judged[sampled_query_index] + 1
            #update the posterior distribution
            posterior_distribution = []
            for query_index in xrange(0, len(queryList)):
                #np.random.seed(0)
                posterior_distribution.append(np.random.beta(query_count[query_index][0], query_retrieved_judged[query_index]))


        previous_pool_budget = budget_tracker
        this_level_pooled_docs_list = copy.deepcopy(pooled_docList)
        pooled_Document[pool] = [this_level_pooled_docs_list, relevant_doc_count]

    return pooled_Document


    for pool, document_list in sorted(pooled_Document.iteritems()):
        print pool, len(document_list[0]), document_list[1], topicPoolInfo[topicId][pool][0], topicPoolInfo[topicId][pool][1]



def initialize_query_level_MTF(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = []  # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = []  # key is the index of query same as queryList and value is how many times it has been sampled so far
    query_retrieved_judged = []
    posterior_distribution = []
    pooled_docList = [] # dummy variable to be used by query_level_bandits function
    docLabel = 0 # dummy variable to be used by query_level_bandits function
    uniqueDoc = 0
    for query_index in xrange(0, len(queryList)):
        relevant_count = 1
        nonrelevant_count = 1
        topicFinished = 0
        query_count.append([relevant_count, nonrelevant_count])
        query_sampled.append(0)  # initially no sampled at all
        query_retrieved_judged.append(2)  # one relevant and one non relevant
        np.random.seed(3) # for initialization
        posterior_distribution.append(pool_value)
    return [query_count, query_sampled, query_retrieved_judged, posterior_distribution,  pooled_docList, docLabel, topicFinished, uniqueDoc]



def initialize_query_level_bandits(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value, buffer_size):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = []  # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = []  # key is the index of query same as queryList and value is how many times it has been sampled so far
    query_retrieved_judged = []
    posterior_distribution = []
    pooled_docList = [] # dummy variable to be used by query_level_bandits function
    docLabel = 0 # dummy variable to be used by query_level_bandits function
    uniqueDoc = 0
    reward_list = [1,0]
    for query_index in xrange(0, len(queryList)):
        relevant_count = 1
        nonrelevant_count = 1
        topicFinished = 0
        query_count.append([relevant_count, nonrelevant_count])
        query_sampled.append(0)  # initially no sampled at all
        query_retrieved_judged.append(relevant_count + nonrelevant_count)  # one relevant and one non relevant
        np.random.seed(3) # for initialization
        #posterior_distribution.append(np.random.beta(query_count[query_index][0],
        #                                             query_retrieved_judged[query_index] - query_count[query_index][0]))
        #posterior_distribution.append((query_count[query_index][0]*1.0)/(query_count[query_index][0]+query_count[query_index][1]))
        posterior_distribution.append(calculated_posterior(reward_list, buffer_size))

    return [query_count, query_sampled, query_retrieved_judged, posterior_distribution,  pooled_docList, docLabel, topicFinished, uniqueDoc,reward_list]



def query_level_MTF(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value, topic_bandit_infos_list):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = topic_bandit_infos_list[0]  # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = topic_bandit_infos_list[1]  # key is the index of query same as queryList and value is how many times it has been sampled so far
    query_retrieved_judged = topic_bandit_infos_list[2]
    posterior_distribution = topic_bandit_infos_list[3]
    pooled_docList = topic_bandit_infos_list[4]
    docLabel = topic_bandit_infos_list[5]
    topicFinished = topic_bandit_infos_list[6] # 0 means not finished
    uniqueDoc = 0

    # go over the number of times each query is sampled
    # if anyof those already sampled 10 times
    # set it posterior to 0.0
    for query_index in xrange(0, len(queryList)):
        if query_sampled[query_index] == pool_value:
            posterior_distribution[query_index] = -1

    # now check if all queries are actuall sampled to pool_values times
    # if so this topicId is exhausted
    if np.sum(posterior_distribution) == -1*len(posterior_distribution):
        topicFinished = 1
        return [query_count, query_sampled,query_retrieved_judged, posterior_distribution, pooled_docList, docLabel, topicFinished, uniqueDoc]

    # that means the topicId is still has some query to investigate
    sampled_query_index = np.argmax(posterior_distribution)

    # get the next available documents for the query
    sampled_query_string = queryList[sampled_query_index]
    # print sampled_query_string,  query_sampled[sampled_query_index]
    sampled_document = topicInfo[topicId][sampled_query_string][query_sampled[sampled_query_index]]
    if sampled_document not in pooled_docList:
        pooled_docList.append(sampled_document)
        docLabel = topic_qrels[str(topicId)][sampled_document]
        uniqueDoc = 1 # it is unique Doc so we judged it and count towards total_budgte
        if docLabel == 0:
            posterior_distribution[sampled_query_index] = posterior_distribution[sampled_query_index] - 1

    # if document is already in docList or not
    # the queryvariants should get credit always
    # increase the pointer of document for the query to the next
    query_sampled[sampled_query_index] = query_sampled[sampled_query_index] + 1
    # get the label
    docLabel = topic_qrels[str(topicId)][sampled_document]
    # update the query_count relevant and non-relevant count

    query_count[sampled_query_index] = [query_count[sampled_query_index][0] + docLabel,
                                        query_count[sampled_query_index][1] + docLabel]
    query_retrieved_judged[sampled_query_index] = query_retrieved_judged[sampled_query_index] + 1

    return [query_count, query_sampled,query_retrieved_judged, posterior_distribution, pooled_docList, docLabel, topicFinished, uniqueDoc]



def query_level_bandits(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value, topic_bandit_infos_list, buffer_size):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = topic_bandit_infos_list[0]  # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = topic_bandit_infos_list[1]  # key is the index of query same as queryList and value is how many times it has been sampled so far
    query_retrieved_judged = topic_bandit_infos_list[2]
    posterior_distribution = topic_bandit_infos_list[3]
    pooled_docList = topic_bandit_infos_list[4]
    docLabel = topic_bandit_infos_list[5]
    topicFinished = topic_bandit_infos_list[6] # 0 means not finished
    reward_list = copy.deepcopy(topic_bandit_infos_list[8])

    uniqueDoc = 0

    # go over the number of times each query is sampled
    # if anyof those already sampled 10 times
    # set it posterior to 0.0
    for query_index in xrange(0, len(queryList)):
        if query_sampled[query_index] == pool_value:
            posterior_distribution[query_index] = 0.0

    # now check if all queries are actuall sampled to pool_values times
    # if so this topicId is exhausted
    if np.sum(posterior_distribution) == 0.0:
        topicFinished = 1
        return [query_count, query_sampled,query_retrieved_judged, posterior_distribution, pooled_docList, docLabel, topicFinished, uniqueDoc]

    # that means the topicId is still has some query to investigate
    sampled_query_index = np.argmax(posterior_distribution)

    # get the next available documents for the query
    sampled_query_string = queryList[sampled_query_index]
    # print sampled_query_string,  query_sampled[sampled_query_index]
    sampled_document = topicInfo[topicId][sampled_query_string][query_sampled[sampled_query_index]]
    if sampled_document not in pooled_docList:
        pooled_docList.append(sampled_document)
        docLabel = topic_qrels[str(topicId)][sampled_document]
        uniqueDoc = 1 # it is unique Doc so we judged it and count towards total_budgte

    # if document is already in docList or not
    # the queryvariants should get credit always
    # increase the pointer of document for the query to the next
    query_sampled[sampled_query_index] = query_sampled[sampled_query_index] + 1
    # get the label
    docLabel = topic_qrels[str(topicId)][sampled_document]
    # update the query_count relevant and non-relevant count

    query_count[sampled_query_index] = [query_count[sampled_query_index][0] + docLabel,
                                        query_count[sampled_query_index][1] + docLabel]
    query_retrieved_judged[sampled_query_index] = query_retrieved_judged[sampled_query_index] + 1

    # update reward_list
    reward_list.append(docLabel)

    # update the posterior distribution
    posterior_distribution[sampled_query_index] = calculated_posterior(reward_list, buffer_size)

    #posterior_distribution[sampled_query_index] = (query_count[sampled_query_index][0]*1.0)/(query_count[sampled_query_index][0]+query_count[sampled_query_index][1])
    return [query_count, query_sampled,query_retrieved_judged, posterior_distribution, pooled_docList, docLabel, topicFinished, uniqueDoc, reward_list]



def topic_level_MTF(topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value, data_path, datasource, systemName):
    topic_budget = {}
    total_budget = 0
    # topic budget is equal to pooled budget level 10 - 1
    np.random.seed(3)
    topic_count = {} # key is the topicId and [relevant count and non relevant count]
    posterior_distribution = []
    topic_index_to_topicID = {}
    topic_index = 0
    topic_all_infos_bandits = {} # topicId --> [query_count, query_sampled, query_retrieved_judged, posterior_distribution]

    for topicId, pooled_budget in sorted(topicPoolInfo.iteritems()):
        topic_budget[topicId] = pooled_budget[pool_depth-1][0] # 0th index is the budget, 10 - 1
        total_budget = total_budget + topic_budget[topicId]
        relevant_count = 1
        nonrelevant_count = 1
        topic_count[topicId] = [relevant_count, nonrelevant_count]
        posterior_distribution.append(1)
        topic_index_to_topicID[topic_index] = topicId
        topic_all_infos_bandits[topicId] = copy.deepcopy(initialize_query_level_MTF(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value))

        topic_index = topic_index + 1

    #budget_list = list(xrange(500, total_budget, 500))
    #budget_list.append(total_budget)

    budget_list = list(xrange(100, 2000, 100))
    budget_list.append(2000)

    # setting all topics priority to max_budgte
    for i in xrange(0, len(posterior_distribution)):
        posterior_distribution[i] = total_budget

    previous_budget = 0

    budget_docList = {} # key is the budget_limit

    for budget_limit in budget_list:
        budget_tracker = previous_budget

        while budget_tracker < budget_limit:
            #print
            sampled_topic_index = np.argmax(posterior_distribution)
            sampled_topicID = topic_index_to_topicID[sampled_topic_index]
            #print budget_tracker, budget_limit, "topicId:", sampled_topicID
            # call query_level_bandit here
            topic_bandit_infos_list = topic_all_infos_bandits[sampled_topicID]
            topicFinished = topic_bandit_infos_list[6]
            if topicFinished == 1:
                print "topicID", sampled_topicID, "Finished"
                posterior_distribution[sampled_topic_index] = -1
                continue
            topic_all_infos_bandits[sampled_topicID] = copy.deepcopy(query_level_MTF(sampled_topicID, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value,
                                topic_bandit_infos_list))

            docLabel = topic_all_infos_bandits[sampled_topicID][5]
            uniqueDoc = topic_all_infos_bandits[sampled_topicID][7]
            topic_count[sampled_topicID][0] =   topic_count[sampled_topicID][0] + docLabel
            topic_count[sampled_topicID][1] =   topic_count[sampled_topicID][1] + docLabel
            total_retrieved = topic_count[sampled_topicID][0] + topic_count[sampled_topicID][1]
            if docLabel == 0: # decrese the priority since it retries a non-releavnt document
                posterior_distribution[sampled_topic_index] = posterior_distribution[sampled_topic_index] - 1
            budget_tracker = budget_tracker + uniqueDoc

        previous_budget = budget_tracker
        topic_to_selectedDocuments = {} # topicId --> list of document
        total_len = 0
        for topicId in sorted(topicPoolInfo.iterkeys()):
            topic_to_selectedDocuments[topicId] = topic_all_infos_bandits[topicId][4]
            total_len = total_len + len(topic_all_infos_bandits[topicId][4])
        budget_docList[budget_limit] = topic_to_selectedDocuments
        print "budget limit:",  budget_limit, "total_len", total_len

        #exit(0)
    filename = data_path + "pooledDocument_" + datasource + "_" + systemName + "_MABNS_Hirarchical.pickle"
    pickle.dump(budget_docList, open(filename, "wb"))

    return budget_docList


def calculated_posterior(reward_array, buffer_size):
    reward_list = None

    if len(reward_array) >= buffer_size:
        reward_list = copy.deepcopy(reward_array[-buffer_size:])
    else:
        reward_list = copy.deepcopy(reward_array)

    # sucesses and failures
    success_count = reward_list.count(1)
    failure_count = len(reward_list) - success_count

    return np.random.beta(1 + success_count, 1 + failure_count)



def topic_level_bandit_ns(topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value, data_path, datasource, systemName, buffer_size):
    topic_budget = {}
    total_budget = 0
    # topic budget is equal to pooled budget level 10 - 1
    np.random.seed(3)
    topic_count = {} # key is the topicId and [relevant count and non relevant count]
    topic_reward_list = {} # key topicId, list is reward list

    posterior_distribution = []
    topic_index_to_topicID = {}
    topic_index = 0
    topic_all_infos_bandits = {} # topicId --> [query_count, query_sampled, query_retrieved_judged, posterior_distribution]

    for topicId, pooled_budget in sorted(topicPoolInfo.iteritems()):
        topic_budget[topicId] = pooled_budget[pool_depth-1][0] # 0th index is the budget, 10 - 1
        total_budget = total_budget + topic_budget[topicId]
        relevant_count = 1.0
        nonrelevant_count = 1.0
        reward_list = [1,0] # initial reward list
        topic_reward_list[topicId] = copy.deepcopy(reward_list)
        topic_count[topicId] = [relevant_count, nonrelevant_count]
        #BLA
        #posterior_distribution.append(np.random.beta(topic_count[topicId][0], topic_count[topicId][1]))

        #MM
        posterior_distribution.append((topic_count[topicId][0]*1.0)/(topic_count[topicId][0] + topic_count[topicId][1]))

        topic_index_to_topicID[topic_index] = topicId
        topic_all_infos_bandits[topicId] = copy.deepcopy(initialize_query_level_bandits(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value, buffer_size))

        topic_index = topic_index + 1

    #budget_list = list(xrange(500, total_budget, 500))
    #budget_list.append(total_budget)

    budget_list = list(xrange(100, 2000, 100))
    budget_list.append(2000)

    previous_budget = 0

    budget_docList = {} # key is the budget_limit

    for budget_limit in budget_list:
        budget_tracker = previous_budget

        while budget_tracker < budget_limit:
            #print
            sampled_topic_index = np.argmax(posterior_distribution)
            sampled_topicID = topic_index_to_topicID[sampled_topic_index]
            #print budget_tracker, budget_limit, "topicId:", sampled_topicID
            # call query_level_bandit here
            topic_bandit_infos_list = topic_all_infos_bandits[sampled_topicID]
            topicFinished = topic_bandit_infos_list[6]
            if topicFinished == 1:
                print "topicID", sampled_topicID, "Finished"
                posterior_distribution[sampled_topic_index] = 0.0
                continue
            topic_all_infos_bandits[sampled_topicID] = copy.deepcopy(query_level_bandits(sampled_topicID, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value,
                                topic_bandit_infos_list,buffer_size))

            docLabel = topic_all_infos_bandits[sampled_topicID][5]
            uniqueDoc = topic_all_infos_bandits[sampled_topicID][7]
            reward_list = copy.deepcopy(topic_reward_list[sampled_topicID])
            reward_list.append(docLabel)

            topic_reward_list[sampled_topicID] = copy.deepcopy(reward_list)



            topic_count[sampled_topicID][0] =   topic_count[sampled_topicID][0] + docLabel
            topic_count[sampled_topicID][1] =   topic_count[sampled_topicID][1] + docLabel
            total_retrieved = topic_count[sampled_topicID][0] + topic_count[sampled_topicID][1]
            #posterior_distribution[sampled_topic_index] = np.random.beta(topic_count[sampled_topicID][0], total_retrieved - topic_count[sampled_topicID][0])
            #MM

            #posterior_distribution[sampled_topic_index]= (topic_count[sampled_topicID][0]*1.0)/(topic_count[sampled_topicID][0] + topic_count[sampled_topicID][1])

            posterior_distribution[sampled_topic_index] = calculated_posterior(reward_list, buffer_size)
            budget_tracker = budget_tracker + uniqueDoc

        previous_budget = budget_tracker
        topic_to_selectedDocuments = {} # topicId --> list of document
        total_len = 0
        for topicId in sorted(topicPoolInfo.iterkeys()):
            topic_to_selectedDocuments[topicId] = topic_all_infos_bandits[topicId][4]
            total_len = total_len + len(topic_all_infos_bandits[topicId][4])
        budget_docList[budget_limit] = topic_to_selectedDocuments
        print "budget limit:",  budget_limit, "total_len", total_len

        #exit(0)
    filename = data_path + "pooledDocument_" + datasource + "_" + systemName + "_MABNS_Hirarchical.pickle"
    pickle.dump(budget_docList, open(filename, "wb"))

    return budget_docList

# topicInfo --> per topic Ranked List of Documents for each query variants
# topic_qrels original qrels file
# topicPoolInf --> pool wise budget info
def query_variants_MTF(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    #queryList = topicSortedInfo[topicId] # it is a list
    poole_budget = topicPoolInfo[topicId]
    max_number_of_samples = pool_value
    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = [] # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = [] # key is the index of query same as queryList and value is how many times it has been sampled so far
    query_prioritiy_list = []
    np.random.seed(3)

    for query_index in xrange(0, len(queryList)):
        relevant_count = 1
        nonrelevant_count = 1
        query_count.append([relevant_count, nonrelevant_count])
        query_sampled.append(0) # initially no sampled at all
        query_prioritiy_list.append(pool_value) # initially all query variants run has equal highest priority to pool_value 10
    previous_pool_budget = 0
    # when pool_depth = 1, pool_docList will conatains all dcoument from pool_1
    # when pool_depth = 2, pool_docList will conatains all dcoument from pool_1 and pool_2
    # when pool_depth = n, pool_docList will conatains all dcoument from pool_1,pool_2, ..., pool_n


    pooled_docList = []
    pooled_Document = {} # key pool and values --> (pooled_docList, relevant_doc_count in that pool_docList)
    relevant_doc_count = 0

    for pool, infoList in sorted(poole_budget.iteritems()):
        total_budget = infoList[0] # per pooled budget
        budget_tracker = previous_pool_budget
        #print "pool:", pool, "total budget", total_budget
        while budget_tracker < total_budget:
            # deterministic MTF
            # because np.max index will find only the first element with match
            #sampled_query_index = query_prioritiy_list.index(np.max(query_prioritiy_list))

            # randomized MTF
            index_list = [index for index, value in enumerate(query_prioritiy_list) if value == np.max(query_prioritiy_list)]
            random_index = np.random.randint(len(index_list), size=1)[0]
            sampled_query_index = index_list[random_index]

            # if any of the query hits the pool_value which is 10 here
            # set its posterior distribution to 0.0
            # so that we will not sample from that query variants
            if query_sampled[sampled_query_index] == pool_value:
                query_prioritiy_list[sampled_query_index] = -1
                continue
            # get the next available documents for the query
            sampled_query_string = queryList[sampled_query_index]
            #print sampled_query_string,  query_sampled[sampled_query_index]
            sampled_document = topicInfo[topicId][sampled_query_string][query_sampled[sampled_query_index]]
            if sampled_document not in pooled_docList:
                pooled_docList.append(sampled_document)
                docLabel = topic_qrels[str(topicId)][sampled_document]
                relevant_doc_count = relevant_doc_count + docLabel
                if docLabel == 0:
                    query_prioritiy_list[sampled_query_index] = query_prioritiy_list[sampled_query_index] - 1
                    #print "Non relevant Doc. Next sampled query should different from before"

                # only update the budget if it is a new document
                budget_tracker = budget_tracker + 1

            # if document is already in docList or not
            # the queryvariants should get credit always
            # increase the pointer of document for the query to the next
            query_sampled[sampled_query_index] = query_sampled[sampled_query_index] + 1



        previous_pool_budget = budget_tracker
        this_level_pooled_docs_list = copy.deepcopy(pooled_docList)
        pooled_Document[pool] = [this_level_pooled_docs_list, relevant_doc_count]

    return pooled_Document

    for pool, document_list in sorted(pooled_Document.iteritems()):
        print pool, len(document_list[0]), document_list[1], topicPoolInfo[topicId][pool][0], topicPoolInfo[topicId][pool][1]



# topicInfo --> per topic Ranked List of Documents for each query variants
# topic_qrels original qrels file
# topicPoolInf --> pool wise budget info
def query_variants_MTF_budget_per_topic(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value, total_budget):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    #queryList = topicSortedInfo[topicId] # it is a list
    poole_budget = topicPoolInfo[topicId]
    max_number_of_samples = pool_value
    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = [] # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = [] # key is the index of query same as queryList and value is how many times it has been sampled so far
    query_prioritiy_list = []
    np.random.seed(3)

    for query_index in xrange(0, len(queryList)):
        relevant_count = 1
        nonrelevant_count = 1
        query_count.append([relevant_count, nonrelevant_count])
        query_sampled.append(0) # initially no sampled at all
        query_prioritiy_list.append(pool_value) # initially all query variants run has equal highest priority to pool_value 10
    previous_pool_budget = 0
    # when pool_depth = 1, pool_docList will conatains all dcoument from pool_1
    # when pool_depth = 2, pool_docList will conatains all dcoument from pool_1 and pool_2
    # when pool_depth = n, pool_docList will conatains all dcoument from pool_1,pool_2, ..., pool_n


    pooled_docList = []
    relevant_doc_count = 0

    budget_tracker = 0
    #print "pool:", pool, "total budget", total_budget
    while budget_tracker < total_budget:

        # check if all query varianst is wxhausted
        if np.sum(query_prioritiy_list) == -1*len(query_prioritiy_list):
            break
        # deterministic MTF
        # because np.max index will find only the first element with match
        #sampled_query_index = query_prioritiy_list.index(np.max(query_prioritiy_list))

        # randomized MTF
        index_list = [index for index, value in enumerate(query_prioritiy_list) if value == np.max(query_prioritiy_list)]
        random_index = np.random.randint(len(index_list), size=1)[0]
        sampled_query_index = index_list[random_index]

        # if any of the query hits the pool_value which is 10 here
        # set its posterior distribution to 0.0
        # so that we will not sample from that query variants
        if query_sampled[sampled_query_index] == pool_value:
            query_prioritiy_list[sampled_query_index] = -1
            continue
        # get the next available documents for the query
        sampled_query_string = queryList[sampled_query_index]
        #print sampled_query_string,  query_sampled[sampled_query_index]
        sampled_document = topicInfo[topicId][sampled_query_string][query_sampled[sampled_query_index]]
        if sampled_document not in pooled_docList:
            pooled_docList.append(sampled_document)
            docLabel = topic_qrels[str(topicId)][sampled_document]
            relevant_doc_count = relevant_doc_count + docLabel
            if docLabel == 0:
                query_prioritiy_list[sampled_query_index] = query_prioritiy_list[sampled_query_index] - 1
                #print "Non relevant Doc. Next sampled query should different from before"

            # only update the budget if it is a new document
            budget_tracker = budget_tracker + 1

        # if document is already in docList or not
        # the queryvariants should get credit always
        # increase the pointer of document for the query to the next
        query_sampled[sampled_query_index] = query_sampled[sampled_query_index] + 1


    return pooled_docList



def query_variants_MTF_budget(topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value, data_path, datasource, systemName, topic_budget_docs):
    total_budget = 0
    topic_budget = {}
    for topicId, pooled_budget in sorted(topicPoolInfo.iteritems()):
        topic_budget[topicId] = pooled_budget[pool_depth - 1][0]  # 0th index is the budget, 10 - 1
        total_budget = total_budget + topic_budget[topicId]

    budget_list = list(xrange(500, total_budget, 500))
    topic_list = list(sorted(topicPoolInfo.keys()))
    noofTopics = len(topic_list)*1.0
    # do not append this since for MTF
    # when total budget is 500, we are passing 500/50.0 = 10
    #budget_list.append(total_budget)

    budget_docList = {}  # key is the budget_limit

    for budget_limit in budget_list:
        budget_per_topic = int(budget_limit/noofTopics)
        topic_to_selectedDocuments = {}  # topicId --> list of document
        total_len = 0
        for topicId in topic_list:
            topic_to_selectedDocuments[topicId] = query_variants_MTF_budget_per_topic(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value,budget_per_topic)
            total_len = total_len + len(topic_to_selectedDocuments[topicId])
            #print topicId, budget_per_topic, len(topic_to_selectedDocuments[topicId])
        print budget_limit, total_len
        budget_docList[budget_limit] = topic_to_selectedDocuments

    # getting all documents for final budget from Bandit Method
    # because we are not running MTF for last budget
    budget_docList[total_budget] = topic_budget_docs[total_budget]
    filename = data_path + "pooledDocument_" + datasource + "_" + systemName + "_MTF_Hirarchical.pickle"
    pickle.dump(budget_docList, open(filename, "wb"))

    return budget_docList

    # topicInfo --> per topic Ranked List of Documents for each query variants
# topic_qrels original qrels file
# topicPoolInf --> pool wise budget info
def query_variants_roundRobin(topicId, topicInfo, topicSortedInfo, topicPoolInfo, topic_qrels, pool_value):
    queryInfo = topicSortedInfo[topicId]
    queryList = list(queryInfo.keys())

    #queryList = topicSortedInfo[topicId]
    poole_budget = topicPoolInfo[topicId]
    max_number_of_samples = pool_value
    # do not sort the queryInfo
    # otherwise it will sort by the query
    # follwoing code confirms that
    '''
    for query, score in queryInfo.iteritems():
        print query, score
    print queryList
    '''
    ##################################

    ## intialize query distribution to uniform distribution
    query_count = [] # key is the index of query same as queryList and [relevant count and non relevant count]
    query_sampled = [] # key is the index of query same as queryList and value is how many times it has been sampled so far

    for query_index in xrange(0, len(queryList)):
        relevant_count = 1
        nonrelevant_count = 1
        query_sampled.append(0) # initially no sampled at all

    previous_pool_budget = 0
    # when pool_depth = 1, pool_docList will conatains all dcoument from pool_1
    # when pool_depth = 2, pool_docList will conatains all dcoument from pool_1 and pool_2
    # when pool_depth = n, pool_docList will conatains all dcoument from pool_1,pool_2, ..., pool_n


    pooled_docList = []
    pooled_Document = {} # key pool and values --> (pooled_docList, relevant_doc_count in that pool_docList)
    relevant_doc_count = 0

    round_robin_counter = 0
    number_of_query_variants = len(queryList)
    for pool, infoList in sorted(poole_budget.iteritems()):
        total_budget = infoList[0] # per pooled budget
        budget_tracker = previous_pool_budget
        #print "pool:", pool, "total budget", total_budget
        while budget_tracker < total_budget:
            sampled_query_index = round_robin_counter%number_of_query_variants
            round_robin_counter = round_robin_counter + 1
            # if any of the query hits the pool_value which is 10 here
            # set its posterior distribution to 0.0
            # so that we will not sample from that query variants
            if query_sampled[sampled_query_index] == pool_value:
                #posterior_distribution[sampled_query_index] = 0.0
                continue
            # get the next available documents for the query
            sampled_query_string = queryList[sampled_query_index]
            #print sampled_query_string,  query_sampled[sampled_query_index]
            sampled_document = topicInfo[topicId][sampled_query_string][query_sampled[sampled_query_index]]
            if sampled_document not in pooled_docList:
                pooled_docList.append(sampled_document)
                docLabel = topic_qrels[str(topicId)][sampled_document]
                relevant_doc_count = relevant_doc_count + docLabel
                # only update the budget if it is a new document
                budget_tracker = budget_tracker + 1

            # if document is already in docList or not
            # the queryvariants should get credit always
            # increase the pointer of document for the query to the next
            query_sampled[sampled_query_index] = query_sampled[sampled_query_index] + 1


        previous_pool_budget = budget_tracker
        this_level_pooled_docs_list = copy.deepcopy(pooled_docList)
        pooled_Document[pool] = [this_level_pooled_docs_list, relevant_doc_count]

    return pooled_Document

    for pool, document_list in sorted(pooled_Document.iteritems()):
        print pool, len(document_list[0]), document_list[1], topicPoolInfo[topicId][pool][0], topicPoolInfo[topicId][pool][1]


def getInfoForAlex(topic_qrels, topicInfo):
    topicQueryRelatedDocs  = {}

    for topicId, queryInfo in sorted(topicInfo.iteritems()):
        queryRelatedDocsInfo = {}
        for queryVariants, docList in sorted(queryInfo.iteritems()):
            print topicId, queryVariants, len(docList)

            relatedDocList = []
            for docId in docList:
                if docId in topic_qrels[str(topicId)]:
                    docLabel = topic_qrels[str(topicId)][docId]
                    if docLabel == 1:
                        relatedDocList.append(docId)

            queryRelatedDocsInfo[queryVariants] = relatedDocList
        topicQueryRelatedDocs[topicId] = queryRelatedDocsInfo

    file_name_alex = data_path + "uqv_qrels_for_alex_" + datasource + "_" + systemName + ".cpickle"

    import pickle
    pickle.dump(topicQueryRelatedDocs, open(file_name_alex, "wb"))

    '''
    topicQueryRelatedDocs = pickle.load(open(file_name_alex, "rb"), encoding="bytes")

    for topicId, queryRelatedDocsInfo in sorted(topicQueryRelatedDocs.iteritems()):
        for queryVariants, docList in sorted(queryRelatedDocsInfo.iteritems()):
            print topicId, queryVariants, len(docList), len(topicInfo[topicId][queryVariants])
    '''

def decode_Alex_results(data_path, datasource):
    filename = data_path + "results_" + datasource + ".csv"
    print filename
    f = open(filename)
    next(f) # skiping the first line
    topic_set_docList = {}
    for lines in f:
        content = lines.split("|")
        topicId = int(content[0])
        set_number = int(content[1])
        query_str = content[2]
        print topicId, set_number, query_str
        set_to_queryList = {}
        if topicId in topic_set_docList:
            set_to_queryList = topic_set_docList[topicId]
            queryList = []
            if set_number in set_to_queryList:
                queryList = set_to_queryList[set_number]
            queryList.append(query_str)
            set_to_queryList[set_number] = queryList
        else:
            queryList = []
            queryList.append(query_str)
            set_to_queryList[set_number] = queryList
            topic_set_docList[topicId] = set_to_queryList

    filename_write = data_path + datasource + "file_query_variants_by_alex_most_popular.pickle"
    pickle.dump(topic_set_docList, open(filename_write, "wb"))

    for topicId, set_to_queryList in sorted(topic_set_docList.iteritems()):
        for set_number, queryList in sorted(set_to_queryList.iteritems()):
            print topicId, set_number, len(queryList)

    return topic_set_docList





if __name__ == '__main__':

    '''
    datasource = 'WT2013'
    start_top = 201
    end_top = 251
    pool_depth = 10
    rankMetric = "map"
    systemName = "user-Indri-BM.uqv.run"
    '''


    datasource = sys.argv[1]
    systemName = sys.argv[2]
    rankMetric = sys.argv[3]
    BUFFER_SIZE = sys.argv[4]

    pool_depth = 10
    start_top = start_topic[datasource]
    end_top = end_topic[datasource]

    # start reading the run files
    systemAddress_file = "/work/04549/mustaf/lonestar/data/collection/ClueWeb12UQV/runs/"

    qrelAddress_file = "/work/04549/mustaf/lonestar/data/collection/ClueWeb12UQV/uqv100-qrels-median-labels.txt"
    data_path = "/work/04549/mustaf/lonestar/data/collection/ClueWeb12UQV/"
    topic_original_qrels_filename = "uqv100-"+datasource


    topicData = TRECTopics(datasource, start_top, end_top)
    topic_qrels = topicData.qrelsReader(qrelAddress_file, data_path, topic_original_qrels_filename)

    topicData_TREC = TRECTopics(datasource, start_top, end_top)
    topic_original_qrels_TREC_filename = "TREC-" + datasource
    topic_qrels_TREC = topicData.qrelsReader(qrelAddress[datasource], data_path, topic_original_qrels_TREC_filename)

    # creating a uqv qrel file which intersects with official qrels
    # labels are also collected from TREC
    s = ""
    uqv_qrels_file_name = data_path + "uqv_qrels_" + datasource + "_" + systemName + ".txt"
    uqv_qrels_topic_relevantDocsCount = {}
    uqv_qrels_topic_relevantDocsCount_file = data_path + datasource + "_" + systemName + "uqv_qrels_topic_relevantDocCounts.pickle"
    for topicId in xrange(start_top, end_top):
        number_of_relevant_docs = 0
        for docId, docLabel in topic_qrels[str(topicId)].iteritems():
            # print (topicId, docId, docLabel)
            s = s + str(topicId) + " 0 " + docId + " " + str(docLabel) + "\n"
            number_of_relevant_docs = number_of_relevant_docs + docLabel
            # check if it is original TREC qrels file
            '''
            if docId in topic_qrels_TREC[str(topicId)]:
                docLabel = topic_qrels_TREC[str(topicId)][docId]
                if docLabel > 1 or docLabel <0:
                    print docLabel
                s = s + str(topicId) + " 0 " + docId + " " + str(docLabel) + "\n"
            '''
        uqv_qrels_topic_relevantDocsCount[topicId] = number_of_relevant_docs

    pickle.dump(uqv_qrels_topic_relevantDocsCount, open(uqv_qrels_topic_relevantDocsCount_file, "wb"))
    f = open(uqv_qrels_file_name, "w")
    f.write(s)
    f.close()

    systemInfoObj = systemReader(datasource, start_top, end_top)
    topicInfo = systemInfoObj.documentListFromUQVVariants(topic_qrels, systemAddress_file, systemName, datasource, data_path)

    topicPooledBudgetInfo = systemInfoObj.getUQVPooledBudgetAndRelevantCount(topicInfo, topic_qrels, pool_depth,
                                                                             data_path, datasource, systemName)

    topicPopularQueryInfoList = systemInfoObj.popularQueryVariantsCountedByNumberOfRelevantDocuments(
        topicInfo, topic_qrels, data_path, datasource, systemName)

    # TopicDiversityInfoSorted is a dictionary TopicNo --> sortedusingDiversityscore({queryName --> diversity_score})
    TopicDiversityInfoSorted = calculate_query_diversity(topicPopularQueryInfoList, data_path, datasource, systemName)




    systemInfoObj_BM25 = systemReader(datasource, start_top, end_top)

    topicInfo_BM25 = systemInfoObj_BM25.documentListFromUQVVariants(topic_qrels, systemAddress_file,
                                                                    "user-Indri-BM.uqv.run", datasource, data_path)
    topicPopularQueryInfoList = systemInfoObj_BM25.popularQueryVariantsCountedByNumberOfRelevantDocuments(
        topicInfo_BM25, topic_qrels, data_path, datasource, systemName)

    #topic_set_queryList = decode_Alex_results(data_path, datasource)

    #generate_alex_query_variant_qrels(50, topic_set_queryList, topicPopularQueryInfoList, topicInfo, "most_popular")

    #exit(0)

    pool_value = 10

    #query_variants_MTF_budget(topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels, pool_value, data_path, datasource, systemName)

    topic_budget_docs_MAB_HIL = topic_level_bandit_ns(topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels, pool_value, data_path, datasource, systemName, BUFFER_SIZE)

    qrel_file_generator_all_HIL(topic_budget_docs_MAB_HIL, "MAB_HIL_"+str(BUFFER_SIZE), data_path, datasource, systemName)

    topic_budget_docs_MTF_HIL = topic_level_MTF(topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels, pool_value, data_path, datasource, systemName)

    qrel_file_generator_all_HIL(topic_budget_docs_MTF_HIL, "DYN_MTF_HIL", data_path, datasource, systemName)

    # the following line each topic has a fixed budget for MTF
    #topic_budget_docs_MTF_HIL = query_variants_MTF_budget(topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels, pool_value,data_path, datasource, systemName, topic_budget_docs_MAB_HIL)



    #getInfoForAlex(topic_qrels, topicInfo)

    exit(0)

    '''
    systemInfoObj_BM25 = systemReader(datasource, start_top, end_top)

    topicInfo_BM25 = systemInfoObj_BM25.documentListFromUQVVariants(topic_qrels, systemAddress_file, "user-Indri-BM.uqv.run", datasource, data_path)
    topicPopularQueryInfoList = systemInfoObj_BM25.popularQueryVariantsCountedByNumberOfRelevantDocuments(topicInfo_BM25, topic_qrels, data_path, datasource, systemName)
    
    import cPickle

    with open('/work/04549/mustaf/lonestar/data/collection/ClueWeb12UQV/'+datasource + '_topic_qrank.pickle', 'r') as f:
        x = cPickle.load(f)

    generate_soumya_query_variant_qrels(50, x,topicPopularQueryInfoList, topicInfo, "most_popular")

    
    '''

    # this part is for UQV analysis
    ###############################################


    random_query_variants_qrels(topicPopularQueryInfoList, topicInfo, data_path, datasource, systemName)

    '''
    
    generate_diverse_query_variant_qrels(TopicDiversityInfoSorted, topicInfo, data_path, datasource, systemName, "most_diverse")
    generate_diverse_query_variant_qrels(TopicDiversityInfoSorted, topicInfo, data_path, datasource, systemName, "least_diverse")

    print "#############################################"
    generate_popular_query_variant_qrels(50, topicPopularQueryInfoList, topicInfo, "most_popular")
    generate_popular_query_variant_qrels(50, topicPopularQueryInfoList, topicInfo, "least_popular")
    

    
    ###############################################
    #  UQV analysis part finished

    '''
    exit(0)
    topicPooledBudgetInfo = systemInfoObj.getUQVPooledBudgetAndRelevantCount(topicInfo, topic_qrels, pool_depth, data_path, datasource, systemName)

    #query_variants_bandit(201, topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels)
    pool_value = 10
    topicBanditPooledDocs = {} # key--> topicID values --> pooledDocList, relevant Count
    topicBanditNSPooledDocs = {}  # key--> topicID values --> pooledDocList, relevant Count
    topicBanditNS2PooledDocs = {}  # key--> topicID values --> pooledDocList, relevant Count

    topicBanditMTFPooledDocs = {}  # key--> topicID values --> pooledDocList, relevant Count
    topicRoundRobinPooledDocs = {}


    topicBanditPooledDocs_filename = data_path + "pooledDocument_" + datasource + "_" + systemName + "_MAB.txt"
    topicBanditNSPooledDocs_filename = data_path + "pooledDocument_" + datasource + "_" + systemName + "_MABNS.txt"
    topicBanditNS2PooledDocs_filename = data_path + "pooledDocument_" + datasource + "_" + systemName + "_MABNS2.txt"
    topicBanditMTFPooledDocs_filename = data_path + "pooledDocument_" + datasource + "_" + systemName + "_MTF.txt"
    topicRoundRobinPooledDocs_filename = data_path + "pooledDocument_" + datasource + "_" + systemName + "_RR.txt"


    '''
    if os.path.exists(topicBanditPooledDocs_filename) and os.path.exists(topicBanditNSPooledDocs_filename) \
            and os.path.exists(topicBanditMTFPooledDocs_filename) and os.path.exists(topicRoundRobinPooledDocs_filename):
        print "ALL File exists"
        topicBanditPooledDocs = pickle.load(open(topicBanditPooledDocs_filename, "rb"))
        topicBanditNSPooledDocs = pickle.load(open(topicBanditNSPooledDocs_filename, "rb"))
        topicBanditNS2PooledDocs = pickle.load(open(topicBanditNS2PooledDocs_filename, "rb"))

        topicBanditMTFPooledDocs = pickle.load(open(topicBanditMTFPooledDocs_filename, "rb"))
        topicRoundRobinPooledDocs = pickle.load(open(topicRoundRobinPooledDocs_filename, "rb"))

    else:
    '''
    for topicId in sorted(topicInfo.iterkeys()):
        #print topicId
        #topicBanditPooledDocs[topicId] = query_variants_bandit(topicId, topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels, pool_value)
        #topicBanditNSPooledDocs[topicId] = query_variants_bandit_ns(topicId, topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels, pool_value)

        topicBanditNS2PooledDocs[topicId] = query_variants_bandit_ns_v1(topicId, topicInfo, TopicDiversityInfoSorted,
                                                                    topicPooledBudgetInfo, topic_qrels, pool_value)

        #topicRoundRobinPooledDocs[topicId] = query_variants_roundRobin(topicId, topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels, pool_value)
        #topicBanditMTFPooledDocs[topicId] = query_variants_MTF(topicId, topicInfo, TopicDiversityInfoSorted, topicPooledBudgetInfo, topic_qrels, pool_value)


    #pickle.dump(topicBanditPooledDocs, open(topicBanditPooledDocs_filename, "wb"))
    #pickle.dump(topicBanditNSPooledDocs, open(topicBanditNSPooledDocs_filename, "wb"))
    pickle.dump(topicBanditNS2PooledDocs, open(topicBanditNS2PooledDocs_filename, "wb"))

    #pickle.dump(topicBanditMTFPooledDocs, open(topicBanditMTFPooledDocs_filename, "wb"))
    #pickle.dump(topicRoundRobinPooledDocs, open(topicRoundRobinPooledDocs_filename, "wb"))

    #qrel_file_generator_all(topicBanditNSPooledDocs, "MABNS", data_path, datasource, systemName)
    qrel_file_generator_all(topicBanditNS2PooledDocs, "MABNS2", data_path, datasource, systemName)

    #qrel_file_generator_all(topicBanditPooledDocs, "MAB", data_path, datasource, systemName)
    #qrel_file_generator_all(topicRoundRobinPooledDocs, "RR", data_path, datasource, systemName)
    #qrel_file_generator_all(topicBanditMTFPooledDocs, "MTF", data_path, datasource, systemName)

    exit(0)


    minNumberOfVarinatsAcrossAllTopics, maxNumberOfVariantsAcrossAllTopics = systemInfoObj.minimunNumberofQueryVariants(topicInfo)
    print minNumberOfVarinatsAcrossAllTopics, maxNumberOfVariantsAcrossAllTopics

    random_query_variants_qrels(topicPopularQueryInfoList, topicInfo, data_path, datasource, systemName)

    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(
        systemNameList[datasource], systemAddress[datasource], uqv_qrels_file_name, rankMetric)

    queryVariantsTauDict = {}
    queryVariantsDropDict = {}

    queryVariantsTauFileName = data_path + "qrels_" + datasource + "_" + systemName + "_queryVariants_Tau.pickle"
    queryVariantsDropFileName = data_path + "qrels_" + datasource + "_" + systemName + "_queryVariants_MaxDrop.pickle"

    numberOfSamples = 5
    for queryVariantsNumber in xrange(0, 21):
        tau_list = []
        drop_list = []
        for sample_number in xrange(1, numberOfSamples + 1):
            pseudo_qrels_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_" + str(
                queryVariantsNumber) + "_sample_number_" + str(sample_number) + ".txt"
            print pseudo_qrels_file_name

            predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(
                systemNameList[datasource], systemAddress[datasource], pseudo_qrels_file_name, rankMetric)

            tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)

            tau_list.append(tau)

            max_drop, bal, chal = drop_calculator(original_system_metric_value_list,
                                                  predicted_system_metric_value_list)

            drop_list.append(max_drop)

            print "number of query variants:", queryVariantsNumber, "sample_number", sample_number, "tau:", tau, "drop:", max_drop

        queryVariantsTauDict[queryVariantsNumber] = tau_list
        queryVariantsDropDict[queryVariantsNumber] = drop_list

        print "queryVarinartNumber", queryVariantsNumber, "mean tau:", mean(tau_list), "mean of max_drop", mean(drop_list)

    pickle.dump(queryVariantsTauDict, open(queryVariantsTauFileName, "wb"))
    pickle.dump(queryVariantsDropDict, open(queryVariantsDropFileName, "wb"))







