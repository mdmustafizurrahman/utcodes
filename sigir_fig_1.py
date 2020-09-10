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
from scipy.stats import pearsonr

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


def pearson_correlation_calc(dict1):
    X = []
    Y = []

    for x, all_y in sorted(dict1.iteritems()):
        for y in all_y:
            X.append(x)
            Y.append(y)

    corr, _ = pearsonr(X,Y)
    return corr

datasource = sys.argv[1]  # can be 'TREC8','gov2', 'WT2013','WT2014'
al_protocol = sys.argv[2]  # 'SAL', 'CAL', # SPL is not there yet
seed_selection_type = sys.argv[3]  # 'IS' only
classifier_name = sys.argv[4]  # "LR", "NR"--> means non-relevant all
collection_size = sys.argv[5]  # 'all', 'qrels' qrels --> means consider documents inseide qrels only
al_classifier = sys.argv[6]  # SVM, RF, NB and LR
start_top = int(sys.argv[7])
end_top = int(sys.argv[8])
rankMetric = sys.argv[9]
excluded_systems_index_list_value = int(sys.argv[10])




data_set_list = ['TREC7','TREC8', 'gov2', 'WT2013', 'WT2014']
plot_type_list = ['tau', 'maximum drop', 'unique number of documents']


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(plot_type_list), figsize=(10,12))
fig.subplots_adjust(hspace=0.9)

protocol_result = {}
plot_location = 1

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

    for plot_type in plot_type_list:
        # subplot starting
        plt.subplot(len(data_set_list), len(plot_type_list), plot_location)

        if plot_type == "tau":
            od = None
            od = collections.OrderedDict(sorted(all_taus_for_system_numbers_across_all_shuffles.items()))
            df = pd.DataFrame.from_dict(od, orient='index')

            system_numbers_list = list(sorted(all_taus_for_system_numbers_across_all_shuffles.iterkeys()))
            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):

            if datasource == "TREC7":
                df = df.drop([60])
                system_numbers_list = system_numbers_list[0:len(system_numbers_list)-1]
            if datasource == "WT2013":
                df = df.drop([50])
                system_numbers_list = system_numbers_list[0:len(system_numbers_list) - 1]
            print df

            plt.boxplot(df)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > 12:
                plt.xlabel("number of systems")
            plt.ylabel("tau")
            plt.ylim([0, 1])
            plt.yticks(np.arange(0.0, 1.01, step=0.1), size = 8)
            plt.xticks(np.arange(1, len(system_numbers_list) + 1), system_numbers_list)
            plt.grid(linestyle='dotted')
            plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(pearson_correlation_calc(all_taus_for_system_numbers_across_all_shuffles))[:6], size=8)

        elif plot_type == "maximum drop":
            od = None
            od = collections.OrderedDict(sorted(all_drops_for_system_numbers_across_all_shuffles.items()))
            df = pd.DataFrame.from_dict(od, orient='index')



            system_numbers_list = list(sorted(all_taus_for_system_numbers_across_all_shuffles.iterkeys()))

            if datasource == "TREC7":
                df = df.drop([60])
                system_numbers_list = system_numbers_list[0:len(system_numbers_list)-1]
            if datasource == "WT2013":
                df = df.drop([50])
                system_numbers_list = system_numbers_list[0:len(system_numbers_list) - 1]
            print df

            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
            plt.boxplot(df)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > 12:

                plt.xlabel("number of systems")
            plt.ylabel("Max Drop")
            plt.ylim([0, 15])
            plt.yticks(np.arange(0, 15, step=2), size=8)

            plt.xticks(np.arange(1, len(system_numbers_list) + 1), system_numbers_list)
            plt.grid(linestyle='dotted')
            plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(pearson_correlation_calc(all_drops_for_system_numbers_across_all_shuffles))[:6], size=8)

        elif plot_type == "unique number of documents":
            od = None
            od = collections.OrderedDict(sorted(all_unique_doc_counts_for_system_numbers_across_all_shuffles.items()))
            df = pd.DataFrame.from_dict(od, orient='index')

            system_numbers_list = list(sorted(all_taus_for_system_numbers_across_all_shuffles.iterkeys()))

            if datasource == "TREC7":
                df = df.drop([60])
                system_numbers_list = system_numbers_list[0:len(system_numbers_list)-1]
            if datasource == "WT2013":
                df = df.drop([50])
                system_numbers_list = system_numbers_list[0:len(system_numbers_list) - 1]
            print df

            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
            plt.boxplot(df)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > 12:
                plt.xlabel("number of systems")
            plt.ylabel("# Unique Docs")
            plt.ylim([1000, 5000])
            plt.yticks(np.arange(0, 5500, step=1000), size=8)
            plt.xticks(np.arange(1, len(system_numbers_list) + 1), system_numbers_list)

            plt.grid(linestyle='dotted')
            plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(pearson_correlation_calc(all_unique_doc_counts_for_system_numbers_across_all_shuffles))[:6], size=8)

        plot_location = plot_location + 1

    #plt.xlabel('% of human judgments', size=16)
    #plt.title(data_set_name_list[data_set_name_index], size=8)
    plt.grid(linestyle='dotted')
    #plt.xticks(x_labels_set, x_labels_set)
    '''
    if datasource == 'TREC7' and classifier_name == 'LR' and plot_type == 'tau':
        plt.legend(loc=1)
    elif datasource == 'TREC8' and classifier_name == 'LR' and plot_type == 'tau':
        plt.legend(loc=1)
    elif plot_type == 'drop' and classifier_name == 'NR':
        plt.legend(loc=1)
    elif plot_type == 'mismatch' and classifier_name == 'NR':
        plt.legend(loc=1)
    else:
        plt.legend(loc=4)
    '''


plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'sigir_fig_1_all.pdf', format='pdf', orientation='landscape')


