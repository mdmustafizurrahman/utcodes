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
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from statistics import mean

from global_definition import *


def func1(x, a, b, c):
    return a * np.exp(-b * x) + c


def func2(x, a, b, c):
    return a*np.exp(-b * x) + c

def all_infos_across_all_shuffle(excluded_systems_tau_list):
    all_taus_for_system_numbers_across_all_shuffles = {}
    group_number_list = []
    for group_number, sample_number_list in sorted(excluded_systems_tau_list.iteritems()):
        #print group_number
        for sample_num, number_shuffle_tau_list in sorted(sample_number_list.iteritems()):
            #print group_number, sample_num
            if group_number in all_taus_for_system_numbers_across_all_shuffles:
                tau_list_tmp = all_taus_for_system_numbers_across_all_shuffles[group_number]
                for tau_value in number_shuffle_tau_list:
                    tau_list_tmp.append(tau_value)
                all_taus_for_system_numbers_across_all_shuffles[group_number] = tau_list_tmp
            else:
                tau_list_tmp = []
                for tau_value in number_shuffle_tau_list:
                    tau_list_tmp.append(tau_value)
                all_taus_for_system_numbers_across_all_shuffles[group_number] = tau_list_tmp
        group_number_list.append(group_number)

    #for group_number, tau_list_tmp in all_taus_for_system_numbers_across_all_shuffles.iteritems():
    #    print group_number, len(tau_list_tmp)
    mean_taus_for_system_numbers_across_all_shuffles = {}
    for group_number, tau_list_tmp in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
        mean_taus_for_system_numbers_across_all_shuffles[group_number] = np.mean(tau_list_tmp)
    return mean_taus_for_system_numbers_across_all_shuffles, group_number_list


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


def merge_dict(dict1, dict2):
    for group_num, sample_number_dict2 in sorted(dict2.iteritems()):
        if group_num in dict1:
            tmp_sample_number_dict1 = dict1[group_num]
            for sample_num, value_list2 in sorted(sample_number_dict2.iteritems()):
                if sample_num in tmp_sample_number_dict1 :
                    tmp_list = tmp_sample_number_dict1[sample_num]
                    for val in value_list2:
                        tmp_list.append(val)
                    tmp_sample_number_dict1[sample_num] = tmp_list
                else:
                    tmp_list = []
                    for val in value_list2:
                        tmp_list.append(val)
                    tmp_sample_number_dict1[sample_num] = tmp_list
            dict1[group_num] = tmp_sample_number_dict1
        else:
            dict1[group_num] = sample_number_dict2




al_classifier = sys.argv[1]  # SVM, RF, NB and LR
collection_size = sys.argv[2]  # 'all', 'qrels' qrels --> means consider documents inseide qrels only
rankMetric = sys.argv[3]

data_set_list = ['TREC7','TREC8', 'gov2', 'WT2013', 'WT2014']
#plot_type_list = ['tau', 'tau ap', 'maximum drop', 'unique number of documents']

plot_type_list = ['tau', 'tau ap', 'unique number of documents']

#plot_type_name = ['tau', 'tau ap', 'Max Drop', '# Unique Rel. Docs']
plot_type_name = ['tau', 'tau ap', '# Unique Rel. Docs']


group_start_number = {}
group_start_number['TREC8'] = [2, 21, 30, 37]
group_start_number['TREC7'] = [2, 21, 30, 37]
group_start_number['gov2'] = [2]
group_start_number['WT2013'] = [2]
group_start_number['WT2014'] = [2]

dataset_taus = {}
dataset_tausap = {}
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


    sample_number_list = [0,1,2,3,4]
    excluded_systems_tau_list = {}
    excluded_systems_tau_ap_list = {}
    excluded_systems_drop_list = {}
    excluded_systems_unique_doc_count_list = {}
    system_numbers_list_labels = []

    # for dataset like TREC8 and TREC7
    if datasource == 'TREC8' or datasource == 'TREC7':
        for sample_num in sample_number_list:
            tau_dict = {}
            tau_ap_dict = {}
            drop_dict = {}
            unique_count_dict = {}

            for start_number in group_start_number[datasource]:
                group_considered_file_name = data_path + "grp_start_number_" + str(
                    start_number) + "_group_considered_sample_number_" + str(
                    sample_num) + "_" + datasource + "_" + rankMetric + ".pickle"

                excluded_systems_tau_drop_uniqueDocs_object = pickle.load(open(group_considered_file_name, "rb"))

                tau_dict.update(excluded_systems_tau_drop_uniqueDocs_object[0])
                tau_ap_dict.update(excluded_systems_tau_drop_uniqueDocs_object[1])
                drop_dict.update(excluded_systems_tau_drop_uniqueDocs_object[2])
                unique_count_dict.update(excluded_systems_tau_drop_uniqueDocs_object[3])

            group_consider_values_updated = [tau_dict, tau_ap_dict, drop_dict,
                                     unique_count_dict]
            group_considered_file_name_updated = data_path + "grp_start_number_" + str(
                1) + "_group_considered_sample_number_" + str(
                sample_num) + "_" + datasource + "_" + rankMetric + ".pickle"

            pickle.dump(group_consider_values_updated, open(group_considered_file_name_updated, "wb"))

    for sample_num in sample_number_list:
        '''
        group_considered_file_name = data_path + "group_considered_sample_number_" + str(
            sample_num) + "_" + datasource + "_" + rankMetric + ".pickle"
        '''
        group_considered_file_name = None
        if datasource == 'gov2' or datasource == 'WT2013' or datasource == 'WT2014':
            group_considered_file_name = data_path + "grp_start_number_" + str(
                group_start_number[datasource][0]) + "_group_considered_sample_number_" + str(
                sample_num) + "_" + datasource + "_" + rankMetric + ".pickle"

        if datasource == 'TREC8' or datasource == 'TREC7':
            group_considered_file_name = data_path + "grp_start_number_" + str(
                1) + "_group_considered_sample_number_" + str(
                sample_num) + "_" + datasource + "_" + rankMetric + ".pickle"

        print group_considered_file_name
        excluded_systems_tau_drop_uniqueDocs_object = pickle.load(open(group_considered_file_name, "rb"))

        merge_dict(excluded_systems_tau_list,excluded_systems_tau_drop_uniqueDocs_object[0])
        merge_dict(excluded_systems_tau_ap_list, excluded_systems_tau_drop_uniqueDocs_object[1])
        merge_dict(excluded_systems_drop_list, excluded_systems_tau_drop_uniqueDocs_object[2])
        merge_dict(excluded_systems_unique_doc_count_list, excluded_systems_tau_drop_uniqueDocs_object[3])
    '''
    for group_num, sample_number_list in sorted(excluded_systems_tau_list.iteritems()):
        print group_num
        for sample_num in sample_number_list.iterkeys():
            print sample_num
    '''

    all_taus_for_system_numbers_across_all_shuffles, system_numbers_list_labels = all_infos_across_all_shuffle(excluded_systems_tau_list)
    all_tau_aps_for_system_numbers_across_all_shuffles, system_numbers_list_labels = all_infos_across_all_shuffle(
        excluded_systems_tau_ap_list)

    all_drops_for_system_numbers_across_all_shuffles, system_numbers_list_labels = all_infos_across_all_shuffle(excluded_systems_drop_list)
    all_unique_doc_counts_for_system_numbers_across_all_shuffles, system_numbers_list_labels = all_infos_across_all_shuffle(excluded_systems_unique_doc_count_list)

    dataset_taus[datasource] =  all_taus_for_system_numbers_across_all_shuffles
    dataset_tausap[datasource] = all_tau_aps_for_system_numbers_across_all_shuffles
    dataset_drop_count[datasource] = all_drops_for_system_numbers_across_all_shuffles
    dataset_unique_count[datasource] = all_unique_doc_counts_for_system_numbers_across_all_shuffles



plot_location = 1

#train_datasets = ['TREC7', 'TREC8', 'WT2013', 'WT2014']
#test_datasets = ['TREC8', 'TREC7', 'WT2014', 'WT2013']

#train_datasets = ['WT2013', 'WT2014']
#test_datasets = ['WT2014', 'WT2013']

train_datasets = ['TREC7', 'TREC8', 'TREC7']
test_datasets = ['TREC8', 'TREC7', 'gov2']


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(train_datasets) , ncols=len(plot_type_list), figsize=(10, 8))
fig.subplots_adjust(hspace=0.9)


# leave one test collection style linear regression
for trainsource, testsource in zip(train_datasets, test_datasets):
    print "type", type
    for plot_name_index, type in enumerate(plot_type_list):

        train_X = []
        train_y = []

        # getting all x and y for linear regression
        raw_data_dictionary = {}
        if type == "tau":
            raw_data_dictionary = dataset_taus[trainsource]
        if type == "maximum drop":
            raw_data_dictionary = dataset_drop_count[trainsource]
        if type == "unique number of documents":
            raw_data_dictionary = dataset_unique_count[trainsource]
        if type == "tau ap":
            raw_data_dictionary = dataset_tausap[trainsource]

        # x gropu number
        # y is means of tau,or tau ap or relCount
        for x, y in sorted(raw_data_dictionary.iteritems()):
            train_X.append(x)
            train_y.append(y)

        system_numbers_list = list(sorted(dataset_taus[testsource].iterkeys()))


        if type == "tau":
            raw_data_dictionary = dataset_taus[testsource]
        if type == "maximum drop":
            raw_data_dictionary = dataset_drop_count[testsource]
        if type == "unique number of documents":
            raw_data_dictionary = dataset_unique_count[testsource]
        if type == "tau ap":
            raw_data_dictionary = dataset_tausap[testsource]

        test_X = []
        test_y = []
        for x,y in sorted(raw_data_dictionary.iteritems()):
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
        poly = PolynomialFeatures(degree = 4)

        y_pred = None
        # Train the model using the training sets
        if type == "maximum drop":
            regr.fit(train_X, train_y)
            # Make predictions using the testing set
            y_pred = regr.predict(test_X)

        if type == "unique number of documents":
            train_X_poly = poly.fit_transform(train_X)
            test_X_poly = poly.fit_transform(test_X)
            regr.fit(train_X_poly, train_y)
            y_pred = regr.predict(test_X_poly)

        if type == "tau" or type == "tau ap":
            popt, pcov = curve_fit(func2, train_X.ravel(), train_y.ravel())
            print 'fit: a=%5.3f, b=%5.3f, c=%5.3f' %tuple(popt)
            y_pred = func2(np.array(test_X), *popt)

        # The coefficients
        #print('Coefficients: \n', regr.coef_)
        # The mean squared error
        print('Mean squared error: %.2f'
              % mean_squared_error(test_y, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f'
              % r2_score(test_y, y_pred))

        plt.subplot(len(train_datasets), len(plot_type_list), plot_location)

        # Plot test outputs
        plt.scatter(test_X, test_y, color='red', s = 1, marker=">")
        #plt.scatter(train_X, train_y, color='red', s = 2)
        plt.plot(test_X, y_pred, color='blue', linewidth=3)

        plt.xticks(system_numbers_list, system_numbers_list, size=3, rotation = 45)
        if type == "tau" or type == "tau ap":
            plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)
            plt.ylim([0,1])
            if type == "tau":
                plt.ylabel(r'$\tau$', size=8)
            elif type == "tau ap":
                plt.ylabel(r'$\tau_{ap}$', size=8)



        if type == "maximum drop":
            plt.ylim([0, 10])
            plt.yticks(np.arange(0, 10, step=1), size=8)
        if type == "unique number of documents":
            plt.ylim([1000, 5000])
            plt.yticks(np.arange(0, 6500, step=1000), size=8)
            plt.ylabel(plot_type_name[plot_name_index])

        if plot_location > ((len(train_datasets)-1)*len(plot_type_list)):
            plt.xlabel("# of groups")
        plt.title("Train=" + datasouce_name_toacronym[trainsource] + "\nTest=" + datasouce_name_toacronym[testsource] + "\n MSE=" + str(mean_squared_error(test_y, y_pred))[:9], size=10)
        plt.tight_layout()
        plt.grid(linestyle='dotted')
        plot_location = plot_location + 1
        print plot_address

#plt.savefig(plot_address + testsource + "_" + type +'_linear.pdf', format='pdf')
    #plt.clf()

plt.savefig(plot_address + 'ICTIR_figure_3_all_'+ rankMetric + '.pdf', format='pdf')
