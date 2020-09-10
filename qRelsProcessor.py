import operator
import time
from tqdm import tqdm
import pickle
import os
import sys
from multiprocessing import Pool as ProcessPool
import itertools
from functools import partial
import scipy.stats as stats
import subprocess
from scipy.stats.stats import kendalltau

import matplotlib as mpl
## agg backend is used to create plot as a .png file
mpl.use('agg')
import matplotlib.pyplot as plt

from global_definition import *

# docIndexToDocId is a list of documentID from TREC collection
def qrelWriter(topicId,topic_complete_qrels_address,topic_predictions_address, predicted_qrels_file_address, docIndexToDocId):
    begin = time.time()

    topic_complete_qrels = pickle.load(open(topic_complete_qrels_address + topicId + '.pickle', 'rb'))
    original_labels = topic_complete_qrels[0]
    predicted_label = topic_complete_qrels[1]

    # merging two dictionaries
    n_relevant = 0
    original_predicted_merged_dict = {}
    for k in sorted(original_labels.iterkeys()):
        original_predicted_merged_dict[k] = original_labels[k]
        if original_labels[k] == 1:
            n_relevant = n_relevant + 1
    for k in sorted(predicted_label.iterkeys()):
        original_predicted_merged_dict[k] = predicted_label[k]
        if predicted_label[k] == 1:
            n_relevant = n_relevant + 1
    #print "Total Relevant in original + predicted :", n_relevant

    topic_all_info_file_name = topic_predictions_address + topicId + ".pickle"
    topic_all_info = pickle.load(open(topic_all_info_file_name, 'rb'))

    # key is the train_per_centage_value
    f1_values = {}
    precision_values = {}
    recall_value = {}
    pooled_document_count_values = {}
    non_pooled_document_count_values = {}

    for k in sorted(topic_all_info.iterkeys()):
        # k is the value from train_per_centage
        start = time.time()
        topicID, f1score, precision, recall, train_index_list, test_index_list, y_pred, pooled_document_count, non_pooled_document_count = topic_all_info[k]
        f1_values[k] = f1score
        precision_values[k] = precision
        recall_value[k] = recall
        pooled_document_count_values[k] = pooled_document_count
        non_pooled_document_count_values[k] = non_pooled_document_count

        # qrel writing
        # format ==
        # topic-id 0 document-id relevance
        # separated by space
        predicted_qrels_file_name = predicted_qrels_file_address + topicId +"_" + str(k) + ".txt"
        number_relevant = 0
        with open(predicted_qrels_file_name, 'w') as f:
            for document_index in train_index_list:
                document_id = docIndexToDocId[document_index]
                label = original_predicted_merged_dict[document_index]
                if label == 1:
                    number_relevant = number_relevant + 1
                f.write(topicId+" 0 "+document_id+" "+str(label)+"\n")

            #print str(k), '# relevant', number_relevant
            #print "f1:", f1score
            # when len(train_index_list) == len(test_index_list)
            # that means everything in the train_list already
            # so we do not need to write the test_index_list values
            if len(train_index_list) == len(test_index_list):
                continue
            for test_index_list_index, document_index in enumerate(test_index_list):
                document_id = docIndexToDocId[document_index]
                label = y_pred[test_index_list_index]
                if label == 1:
                    number_relevant = number_relevant + 1
                f.write(topicId + " 0 " + document_id + " " + str(label)+"\n")
            #print str(k), '# relevant', number_relevant
        #print "Finished ", str(k), ": ", time.time() - start

    #print "Finished ", topicId, ": ", time.time() - begin
    return (topicId, f1_values, precision_values, recall_value, pooled_document_count_values, non_pooled_document_count_values)


def qrelWriter_multiprocessing(topic_list,topic_complete_qrels_address,topic_predictions_address, predicted_qrels_file_address, docIndexToDocId, topic_summary_info_file_name):
    num_workers = None
    workers = ProcessPool(num_workers)
    f1_dict = {} # 0,1,2...,10 keys
    precision_dict = {}
    recall_dict = {}
    pooled_count = {}
    non_pooled_count = {}

    file_complete_path = topic_summary_info_file_name + ".pickle"
    # if file already exist, just load it
    if os.path.isfile(file_complete_path):
        print file_complete_path + " exists. Loading from that."
        topic_summary_info = pickle.load(open(file_complete_path, 'rb'))
    else:
        with tqdm(total=len(topic_list)) as pbar:
            partial_qrelWriter = partial(qrelWriter, topic_complete_qrels_address=topic_complete_qrels_address,topic_predictions_address=topic_predictions_address,predicted_qrels_file_address=predicted_qrels_file_address, docIndexToDocId=docIndexToDocId)
            for topic_all_info in tqdm(workers.imap_unordered(partial_qrelWriter, topic_list)):
                topicId = topic_all_info[0]  # 0 is the first tuple

                f1_values = topic_all_info[1]
                f1_dict = update_dict_key_wise(f1_dict,f1_values)

                precision_values = topic_all_info[2]
                precision_dict  = update_dict_key_wise(precision_dict, precision_values)

                recall_values = topic_all_info[3]
                recall_dict = update_dict_key_wise(recall_dict, recall_values)

                pooled_document_count_values = topic_all_info[4]
                pooled_count = update_dict_key_wise(pooled_count, pooled_document_count_values)

                non_pooled_document_count_values = topic_all_info[5]
                non_pooled_count = update_dict_key_wise(non_pooled_count, non_pooled_document_count_values)

                pbar.update()
        file_complete_path = topic_summary_info_file_name + ".pickle"
        topic_summary_info = (f1_dict,precision_dict,recall_dict,pooled_count,non_pooled_count)
        pickle.dump(topic_summary_info, open(file_complete_path, 'wb'))
    return topic_summary_info


def calculateSystemRanks(systemName, systemAddress, relevanceJudgementAddress, rankMetric):
    system = systemAddress + systemName

    #print self.systemAddress, system
    #print systemAddress
    #print relevanceJudgementAddress
    #print systemName
    shellCommand = trec_eval_executable+' -m '+rankMetric+' ' + relevanceJudgementAddress + ' ' + system
    #print shellCommand
    p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    rankMetricValue = 0
    for line in p.stdout.readlines():
        #print line
        values = line.split()
        #print values
        rankMetricValue = float(values[2])
    retval = p.wait()

    return (systemName, rankMetricValue)

def calculateSystemRanks_multiprocessing(systemList, systemAddress, relevanceJudgementAddress, rankMetric):
    num_workers = None
    workers = ProcessPool(processes=20)
    system_metric_value = {} # key is the system_name
    #print systemList
    with tqdm(total=len(systemList)) as pbar:
        partial_calculateSystemRanks = partial(calculateSystemRanks, systemAddress=systemAddress, relevanceJudgementAddress=relevanceJudgementAddress, rankMetric=rankMetric)
        for system_info in tqdm(workers.imap_unordered(partial_calculateSystemRanks, systemList)):
            system_name = system_info[0]
            system_metric_val = system_info[1]
            system_metric_value[system_name] = system_metric_val
            pbar.update()
    workers.close()
    workers.join()
    system_metric_value_list = []
    for system_name in sorted(system_metric_value.iterkeys()):
        system_metric_value_list.append(system_metric_value[system_name])
    return (system_metric_value, system_metric_value_list)


# drop calculator
def drop_calculator(original_list, predicted_list):
    # create a dictionary from original_list and predicted list
    # key is the index of systems
    original_Map = {}
    predicTed_Map = {}

    for i, value in enumerate(original_list):
        original_Map[i] = value

    i = 0
    for i, value in enumerate(predicted_list):
        predicTed_Map[i] = value

    # sort both dictionary by value
    # output is a list of tuple (indexofsystem, value)
    sorted_original_Map = sorted(original_Map.items(), key=operator.itemgetter(1))
    sorted_predicted_Map = sorted(predicTed_Map.items(), key=operator.itemgetter(1))


    original_rank_list = range(len(sorted_original_Map))
    predicted_rank_list = []

    max_drop = 0
    drop_in_rank_list = [] # contains the drop in rank for each system
    delta_in_score_list = [] # contains the corresponsing delta in score for a system
    for rank_in_original_list, x in enumerate(sorted_original_Map):
        # find that system in predicted list along with it ranks
            for rank_in_predicted_list, y in enumerate(sorted_predicted_Map):
                if y[0] == x[0]: # is system id match
                    predicted_rank_list.append(rank_in_predicted_list)
                    drop_in_rank_list.append(abs(rank_in_predicted_list - rank_in_original_list))
                    delta_in_score_list.append(y[1] - x[1]) # difference in score of an system in a ranked list
                    if abs(rank_in_predicted_list - rank_in_original_list) > max_drop:
                        max_drop = abs(rank_in_predicted_list - rank_in_original_list)
                    break
    #print predicted_rank_list

    #return max_drop, drop_in_rank_list, delta_in_score_list
    return max_drop, drop_in_rank_list, delta_in_score_list

    #tau_ap_mine(original_rank_list, predicted_rank_list)





if __name__ == '__main__':

    '''
    import os
    os.chdir('/work/04549/mustaf/maverick/data/TREC/TREC8/')
    for f in os.listdir('.'):
        if f.endswith('.pickle'):
            #if f.startswith('per_topic_complete_'):
                #print f, f.replace('_TREC8_', '_')
            #    os.rename(f, f.replace('_TREC8_', '_'))
            if f.startswith('per_topic_predictions_IS_'):
                os.rename(f, f.replace('_TREC8_', '_'))


    exit(0)
    '''

    datasource = sys.argv[1]  # can be 'TREC8','gov2', 'WT2013','WT2014'
    al_protocol = sys.argv[2]  # 'SAL', 'CAL', # SPL is not there yet
    seed_selection_type = sys.argv[3] # 'IS' only
    classifier_name = sys.argv[4] # "LR", "NR"--> means non-relevant all
    rankMetric = sys.argv[5] # map, infAP
    al_classifier = sys.argv[6] # NB, LR
    use_original_qrels = int(sys.argv[7])  # 1 means use original qrels, other value 0 means
    varied_qrels_directory_number = int(sys.argv[8])  # 1,2,3

    source_file_path =  base_address + datasource + "/"
    data_path = base_address + datasource + "/result/" + al_classifier + "/"

    qrelAddress_path = None
    system_name_list = None
    if use_original_qrels == 1:
        qrelAddress_path = qrelAddress[datasource]
        system_name_list = systemNameList[datasource]
    else:
        data_path = data_path + "varied_pool_" + str(varied_qrels_directory_number) + "/"
        qrelAddress_path = data_path + "relevance.txt"
        k = 0
        excluded_systems = {}
        for i in xrange(0, 45, 5):
            excluded_systems_list = []
            for j in xrange(i, i + 20, 1):
                excluded_systems_list.append(system_runs_TREC8_list[j])
            excluded_systems[k] = excluded_systems_list
            k = k + 1
        system_name_list = excluded_systems[0] # we are using 0th index list

    # contains the text file generated from the pickle file
    predicted_qrels_file_path = data_path + "predictedQrels/"

    print "qrel address path", qrelAddress_path
    print "source_file_path", source_file_path
    print "data_path", data_path

    topic_list = [str(topicID) for topicID in xrange(start_topic[datasource], end_topic[datasource])]

    # loading metadata about the whole document collection
    # basically we are loading one list and one dictionary
    # list is a list of documentID from TREC which is docIndexToDocId
    # dictionary is a map from documentID to documentIndex, which is docIdToDocIndex
    metadata = pickle.load(open(source_file_path + meta_data_file_name[datasource], 'rb'))
    docIndexToDocId = metadata['docIndexToDocId']
    docIdToDocIndex = metadata['docIdToDocIndex']

    # per_topic_complete_qrels address loader
    # it contains original_qrels and predicted_qrels
    # just need to append the topicId to get the file
    topic_complete_qrels_address = data_path + "per_topic_complete_qrels_" + classifier_name + "_"

    # per_topic_prediction contains
    # train_per_centage_data_point, train_list, test_list, pred for index in test_list
    topic_predictions_address = data_path + "per_topic_predictions_" + seed_selection_type + "_" + classifier_name + "_" + al_protocol + "_"
    predicted_qrels_file_address = predicted_qrels_file_path + seed_selection_type + "_" + classifier_name + "_" + al_protocol + "_"

    #topic_list = [str(topicID) for topicID in xrange(start_topic[datasource], end_topic[datasource])]
    topic_summary_info_file_name = data_path + "per_topic_summary_" + seed_selection_type + "_" + classifier_name + "_" + al_protocol
    #topic_list = '402'
    #print topic_list
    topic_summary_info = qrelWriter_multiprocessing(topic_list, topic_complete_qrels_address, topic_predictions_address, predicted_qrels_file_address, docIndexToDocId, topic_summary_info_file_name)

    #topic_summary_info = pickle.load(open(topic_summary_info_file_name+'.pickle','rb'))

    for k, v in topic_summary_info[0].iteritems():
        f1 = topic_summary_info[0][k]/50.0
        precision = topic_summary_info[1][k]/50.0
        recall = topic_summary_info[2][k]/50.0
        pooled_count = topic_summary_info[3][k]/50.0
        non_pooled_count = topic_summary_info[4][k]/50.0
        print k, f1, precision, recall, pooled_count, non_pooled_count

    # do it just for the first time
    # merging per topic wise qrels files into one large file containing qrels file
    # for all 50 topics
    for i in xrange(0, len(train_per_centage)):
        command = 'cat '+ predicted_qrels_file_path + seed_selection_type +'_'+classifier_name+'_'+ al_protocol +'_[0-9]*_' + str(i) + '.txt > '+ predicted_qrels_file_path+ seed_selection_type +'_'+classifier_name+'_'+ al_protocol +'_all_' + str(i) + '.txt'
        print command
        os.system(command)

    
    tau_list = []

    original_system_metric_value, original_system_metric_value_list = calculateSystemRanks_multiprocessing(system_name_list, systemAddress[datasource], qrelAddress_path, rankMetric)
    original_system_metric_value_file_name = data_path + seed_selection_type +'_'+classifier_name+'_'+ al_protocol +'_original_'+ rankMetric+'.pickle'
    pickle.dump(original_system_metric_value, open(original_system_metric_value_file_name, 'wb'))
    drop_list = []
    delta_score_list = []
    for i in xrange(1, len(train_per_centage)):
        relevanceJudgementAddress = predicted_qrels_file_path+ seed_selection_type +'_'+classifier_name+'_'+ al_protocol +'_all_' + str(i) + '.txt'
        predicted_system_metric_value, predicted_system_metric_value_list = calculateSystemRanks_multiprocessing(system_name_list, systemAddress[datasource], relevanceJudgementAddress, rankMetric)
        predicted_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_predicted_' + rankMetric +'_'+str(i) +'.pickle'
        pickle.dump(predicted_system_metric_value, open(predicted_system_metric_value_file_name, 'wb'))
        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)
        tau_list.append(tau)
        drop_in_rank_list, delta_in_score_list = drop_calculator(original_system_metric_value_list, predicted_system_metric_value_list, i)
        drop_list.append(drop_in_rank_list)
        delta_score_list.append(delta_in_score_list)
        print i, tau

    drop_list_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_droplist.pickle'
    pickle.dump(drop_list, open(drop_list_file_name, 'wb'))

    delta_list_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_deltalist.pickle'
    pickle.dump(delta_score_list, open(delta_list_file_name, 'wb'))

    drop_list_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_droplist.pickle'
    delta_list_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_deltalist.pickle'

    drop_list = pickle.load(open(drop_list_file_name, 'rb'))
    delta_score_list = pickle.load(open(delta_list_file_name, 'rb'))

    x_labels = [1,2,3,4,5,6,7,8,9,10]
    x_labels_set = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(drop_list)
    plt.xticks(x_labels, x_labels_set)
    plt.xlabel("% of human judgments")
    plt.ylabel("drop in system rank position")
    plt.title(datasource)
    plt.title("collection = " + datasource + ", collection size = " + str(collection_size[
                                                                              datasource]) + "\n" + "AL classifier = " + al_classifier + ", document selection method = " + al_protocol + "\n rank metric = " + rankMetric )

    drop_fig_name = datasource+"_"+al_classifier+"_"+al_protocol+"_"+rankMetric
    # Save the figure
    fig.savefig(data_path + drop_fig_name + '_drop_rank.png', bbox_inches='tight')
    fig.clear()

    # Create a figure instance
    fig1 = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig1.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(delta_score_list)
    plt.xticks(x_labels, x_labels_set)
    plt.xlabel("% of human judgments")
    plt.ylabel("delta in system ranking score")
    plt.title("collection = " + datasource + ", collection size = " + str(collection_size[
                                                                              datasource]) + "\n" + "AL classifier = " + al_classifier + ", document selection method = " + al_protocol + "\n rank metric = " + rankMetric)

    # Save the figure
    fig.savefig(data_path + drop_fig_name + '_delta_score.png', bbox_inches='tight')

    '''
    
    original_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_original_' + rankMetric + '.pickle'
    original_system_metric_value = pickle.load(open(original_system_metric_value_file_name, 'rb'))
    original_system_metric_value_list = []
    for system_name in sorted(original_system_metric_value.iterkeys()):
        original_system_metric_value_list.append(original_system_metric_value[system_name])

    for i in xrange(1, len(train_per_centage)):
        predicted_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_predicted_' + rankMetric +'_'+str(i) +'.pickle'
        predicted_system_metric_value = pickle.load(open(predicted_system_metric_value_file_name, 'rb'))
        predicted_system_metric_value_list = []
        for system_name in sorted(predicted_system_metric_value.iterkeys()):
            predicted_system_metric_value_list.append(predicted_system_metric_value[system_name])

        tau, p_value = stats.kendalltau(original_system_metric_value_list, predicted_system_metric_value_list)
        tau_list.append(tau)
        print i, tau
    '''


    tau_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_tau_' + rankMetric + '.pickle'
    pickle.dump(tau_list, open(tau_file_name, 'wb'))




