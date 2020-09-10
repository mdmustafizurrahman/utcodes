import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)
import math
import pickle
import sys
import operator
import pandas as pd
import collections
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


from global_definition import *


def all_infos_across_all_shuffle(excluded_systems_tau_list):
    all_taus_for_system_numbers_across_all_shuffles = {}
    for excluded_systems_index, system_numbers_tau_list in excluded_systems_tau_list.iteritems():
        system_numbers_list = []
        means = []
        stds = []
        mins = []
        maxs = []
        for system_numbers, number_shuffle_tau_list in sorted(system_numbers_tau_list.iteritems()):
            system_numbers_list.append(system_numbers)
            #print system_numbers, np.mean(number_shuffle_tau_list)
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


    return all_taus_for_system_numbers_across_all_shuffles


def drop_calculator(original_Map, predicTed_Map):
    # original_Map and predicTed_Map are dictionaries
    # key is the index of systems

    # sort both dictionary by value
    # output is a list of tuple (indexofsystem, value)
    sorted_original_Map = sorted(original_Map.items(), key=operator.itemgetter(1))
    sorted_predicted_Map = sorted(predicTed_Map.items(), key=operator.itemgetter(1))

    original_rank_list = range(len(sorted_original_Map))
    predicted_rank_list = []

    max_diff = 0
    number_of_system_rank_position_mismatch = 0
    for rank_in_original_list, x in enumerate(sorted_original_Map):
        # find that system in predicted list along with it ranks
            for rank_in_predicted_list, y in enumerate(sorted_predicted_Map):
                if y[0] == x[0]: # is system id match
                    predicted_rank_list.append(rank_in_predicted_list)
                    if abs(rank_in_predicted_list - rank_in_original_list) > max_diff:
                        max_diff = abs(rank_in_predicted_list - rank_in_original_list)
                    if rank_in_predicted_list != rank_in_original_list:
                        number_of_system_rank_position_mismatch += 1
                    break

    #print predicted_rank_list

    #return max_diff, number_of_system_rank_position_mismatch, tau_ap_mine(original_rank_list, predicted_rank_list)
    return max_diff, number_of_system_rank_position_mismatch



al_classifier = sys.argv[1]  # SVM, RF, NB and LR
collection_size = sys.argv[2]  # 'all', 'qrels' qrels --> means consider documents inseide qrels only
type = sys.argv[3]
'''
start_top = int(sys.argv[7])
end_top = int(sys.argv[8])
rankMetric = sys.argv[9]
excluded_systems_index_list_value = int(sys.argv[10])
datasource = sys.argv[1]  # can be 'TREC8','gov2', 'WT2013','WT2014'
al_protocol = sys.argv[2]  # 'SAL', 'CAL', # SPL is not there yet
seed_selection_type = sys.argv[3]  # 'IS' only
classifier_name = sys.argv[4]  # "LR", "NR"--> means non-relevant all

'''


data_set_list = ['TREC7','TREC8', 'gov2', 'WT2013', 'WT2014']
plot_type_list = ['tau', 'maximum drop', 'unique number of documents']


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(plot_type_list) , ncols=len(data_set_list), figsize=(10, 8))
fig.subplots_adjust(hspace=0.9)

protocol_result = {}
plot_location = 1

dataset_taus = {}
dataset_unique_count = {}
dataset_drop_count = {}


for data_set_name_index, datasource in enumerate(data_set_list):
    source_file_path = base_address + datasource + "/"
    data_path = base_address + datasource + "/result/" + al_classifier + "/"


    if collection_size == 'qrels':
        source_file_path = base_address + datasource + "/sparseTRECqrels/"
        data_path = base_address + datasource + "/sparseTRECqrels/" + "result/"  + al_classifier + "/"

    print "source_file_path", source_file_path
    print "data_path", data_path

    topic_list = [str(topicID) for topicID in xrange(start_topic[datasource], end_topic[datasource])]
    number_of_topic = len(topic_list)


    excluded_systems_index_list = [0,1,2,3]
    excluded_systems_tau_list = {}
    excluded_systems_drop_list = {}
    excluded_systems_unique_doc_count_list = {}
    system_numbers_list_labels = []

    for excluded_systems_index in excluded_systems_index_list:
        excluded_systems_tau_drop_uniqueDocs_file_name = data_path + datasource + "_pseudo_qrels_all_taus_drop_unique_counts_" + str(
            excluded_systems_index_list[excluded_systems_index]) + ".pickle"

        excluded_systems_tau_drop_uniqueDocs_object = pickle.load(open(excluded_systems_tau_drop_uniqueDocs_file_name, "rb"))
        excluded_systems_tau_list.update(excluded_systems_tau_drop_uniqueDocs_object[0])
        excluded_systems_drop_list.update(excluded_systems_tau_drop_uniqueDocs_object[1])
        excluded_systems_unique_doc_count_list.update(excluded_systems_tau_drop_uniqueDocs_object[2])
        system_numbers_list_labels = excluded_systems_tau_drop_uniqueDocs_object[3]

    all_taus_for_system_numbers_across_all_shuffles = all_infos_across_all_shuffle(excluded_systems_tau_list)
    all_drops_for_system_numbers_across_all_shuffles = all_infos_across_all_shuffle(excluded_systems_drop_list)
    all_unique_doc_counts_for_system_numbers_across_all_shuffles = all_infos_across_all_shuffle(excluded_systems_unique_doc_count_list)

    dataset_taus[datasource] =  all_taus_for_system_numbers_across_all_shuffles
    dataset_drop_count[datasource] = all_drops_for_system_numbers_across_all_shuffles
    dataset_unique_count[datasource] = all_unique_doc_counts_for_system_numbers_across_all_shuffles

plot_location = 1

# leave one test collection style linear regression
for type in plot_type_list:
    print "type", type
    for datasource1 in data_set_list:
        testsource = ""
        trainsource = []
        for datasource2 in data_set_list:
            if datasource1 == datasource2:
                testsource = datasource2
            else:
                trainsource.append(datasource2)


        train_X = []
        train_y = []

        # getting all x and y for linear regression
        raw_data_dictionary = {}
        for source in trainsource:
            if type == "tau":
                raw_data_dictionary = dataset_taus[source]
            if type == "maximum drop":
                raw_data_dictionary = dataset_drop_count[source]
            if type == "unique number of documents":
                raw_data_dictionary = dataset_unique_count[source]

            for x in sorted(raw_data_dictionary.iterkeys()):
                for y in raw_data_dictionary[x]:
                    train_X.append(x)
                    train_y.append(y)
        system_numbers_list = list(sorted(dataset_taus[testsource].iterkeys()))


        if type == "tau":
            raw_data_dictionary = dataset_taus[testsource]
        if type == "maximum drop":
            raw_data_dictionary = dataset_drop_count[testsource]
        if type == "unique number of documents":
            raw_data_dictionary = dataset_unique_count[testsource]

        test_X = []
        test_y = []
        for x in sorted(raw_data_dictionary.iterkeys()):
            for y in raw_data_dictionary[x]:
                test_X.append(x)
                test_y.append(y)

        print "test source", testsource, trainsource
        print "train_set_len", len(train_y)
        print "test_set_len", len(test_y)

        train_X = np.array(train_X).reshape(-1, 1)
        train_y = np.array(train_y).reshape(-1, 1)

        test_X = np.array(test_X).reshape(-1, 1)
        test_y = np.array(test_y).reshape(-1, 1)


        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(train_X, train_y)

        # Make predictions using the testing set
        y_pred = regr.predict(test_X)

        # The coefficients
        print('Coefficients: \n', regr.coef_)
        # The mean squared error
        print('Mean squared error: %.2f'
              % mean_squared_error(test_y, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f'
              % r2_score(test_y, y_pred))

        plt.subplot(len(plot_type_list), len(data_set_list), plot_location)

        # Plot test outputs
        plt.scatter(test_X, test_y, color='black')
        plt.plot(test_X, y_pred, color='blue', linewidth=3)

        plt.xticks(system_numbers_list, system_numbers_list, size=5)
        if type == "tau":
            plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)

        if type == "maximum drop":
            plt.ylim([0, 15])
            plt.yticks(np.arange(0, 15, step=2), size=8)
        if type == "unique number of documents":
            plt.ylim([1000, 5000])
            plt.yticks(np.arange(0, 5500, step=1000), size=8)

        if plot_location == 1:
            plt.ylabel("tau")
        if plot_location == 6:
            plt.ylabel("Max Drop")
        if plot_location == 11:
            plt.ylabel("# Unique Docs")
        if plot_location >=11:
            plt.xlabel("# of systems")
        plt.title("Test=" + testsource + "\n MSE=" + str(mean_squared_error(test_y, y_pred))[:8], size=5)
        plt.tight_layout()
        plt.grid(linestyle='dotted')
        plot_location = plot_location + 1
        print plot_address

#plt.savefig(plot_address + testsource + "_" + type +'_linear.pdf', format='pdf')
    #plt.clf()

plt.savefig(plot_address + 'linear_regs_all.pdf', format='pdf')
