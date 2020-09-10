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
import numpy as np
from scipy.stats import pearsonr

from global_definition import *


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

    return all_taus_for_system_numbers_across_all_shuffles, group_number_list



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


def pearson_correlation_calc(dict1):
    X = []
    Y = []

    for x, all_y in sorted(dict1.iteritems()):
        for y in all_y:
            X.append(x)
            Y.append(y)

    corr, _ = pearsonr(X,Y)
    return corr

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
rankMetric = sys.argv[3]

'''
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
'''

group_start_number = {}
group_start_number['TREC8'] = [2, 21, 30]
group_start_number['TREC7'] = [2, 21, 30]
group_start_number['gov2'] = [2]
group_start_number['WT2013'] = [2]
group_start_number['WT2014'] = [2]

data_set_list = ['TREC7','TREC8', 'gov2', 'WT2013', 'WT2014']
#data_set_list = ['WT2014', 'WT2013']

#data_set_list = ['TREC7', 'gov2']
data_set_name_list = ['Adhoc\'98', 'Adhoc\'99','TB\'06', 'WT\'13', 'WT\'14']

#plot_type_list = ['tau', 'tau ap', 'maximum drop', 'unique number of documents']
plot_type_list = ['tau', 'tau ap', 'unique number of documents']


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(plot_type_list), figsize=(10,12))
fig.subplots_adjust(hspace=0.9)

protocol_result = {}
plot_location = 1
onlyautomatic = 1

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
                    sample_num) + "_" + datasource + "_" + rankMetric + "_" + str(onlyautomatic) + ".pickle"

                excluded_systems_tau_drop_uniqueDocs_object = pickle.load(open(group_considered_file_name, "rb"))

                tau_dict.update(excluded_systems_tau_drop_uniqueDocs_object[0])
                tau_ap_dict.update(excluded_systems_tau_drop_uniqueDocs_object[1])
                drop_dict.update(excluded_systems_tau_drop_uniqueDocs_object[2])
                unique_count_dict.update(excluded_systems_tau_drop_uniqueDocs_object[3])

            group_consider_values_updated = [tau_dict, tau_ap_dict, drop_dict,
                                     unique_count_dict]
            group_considered_file_name_updated = data_path + "grp_start_number_" + str(
                1) + "_group_considered_sample_number_" + str(
                sample_num) + "_" + datasource + "_" + rankMetric + "_" + str(onlyautomatic) + ".pickle"

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
                sample_num) + "_" + datasource + "_" + rankMetric + "_" + str(onlyautomatic) + ".pickle"

        if datasource == 'TREC8' or datasource == 'TREC7':
            group_considered_file_name = data_path + "grp_start_number_" + str(
                1) + "_group_considered_sample_number_" + str(
                sample_num) + "_" + datasource + "_" + rankMetric + "_" + str(onlyautomatic) + ".pickle"

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

    for plot_type in plot_type_list:
        # subplot starting
        plt.subplot(len(data_set_list), len(plot_type_list), plot_location)

        if plot_type == "tau":
            od = None
            od = collections.OrderedDict(sorted(all_taus_for_system_numbers_across_all_shuffles.items()))
            df = pd.DataFrame.from_dict(od, orient='index')

            system_numbers_list = list(sorted(all_taus_for_system_numbers_across_all_shuffles.iterkeys()))
            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
            print df

            data_frame = []
            for group_no, all_group_infos in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
                tmp_grp_list = []
                for info in all_group_infos:
                    if np.isnan(info) == False:
                        tmp_grp_list.append(info)
                data_frame.append(tmp_grp_list)

            if datasource == 'TREC8' or datasource == 'TREC7':
                data_frame, system_numbers_list = update_dataframe(data_frame, datasource)

            plt.boxplot(data_frame)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > ((len(data_set_list) - 1) * len(plot_type_list)):
                plt.xlabel("number of groups")
            plt.ylabel(r'$\tau$', size=8)
            plt.ylim([0, 1])
            plt.yticks(np.arange(0.0, 1.01, step=0.1), size = 8)
            plt.xticks(np.arange(1, len(system_numbers_list) + 1), system_numbers_list, size = 6)
            plt.grid(linestyle='dotted')
            plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(
                pearson_correlation_calc(all_taus_for_system_numbers_across_all_shuffles))[:6], size=8)

        elif plot_type == "tau ap":
            od = None
            od = collections.OrderedDict(sorted(all_tau_aps_for_system_numbers_across_all_shuffles.items()))
            df = pd.DataFrame.from_dict(od, orient='index')

            system_numbers_list = list(sorted(all_taus_for_system_numbers_across_all_shuffles.iterkeys()))
            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
            print df

            data_frame = []
            for group_no, all_group_infos in sorted(all_tau_aps_for_system_numbers_across_all_shuffles.iteritems()):
                tmp_grp_list = []
                for info in all_group_infos:
                    if np.isnan(info) == False:
                        tmp_grp_list.append(info)
                data_frame.append(tmp_grp_list)

            if datasource == 'TREC8' or datasource == 'TREC7':
                data_frame, system_numbers_list = update_dataframe(data_frame, datasource)

            plt.boxplot(data_frame)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > ((len(data_set_list) - 1) * len(plot_type_list)):
                plt.xlabel("number of groups")
            plt.ylabel(r'$\tau_{ap}$', size=8)
            plt.ylim([0, 1])
            plt.yticks(np.arange(0.0, 1.01, step=0.1), size = 8)
            plt.xticks(np.arange(1, len(system_numbers_list) + 1), system_numbers_list , size = 6)
            plt.grid(linestyle='dotted')
            plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(
                pearson_correlation_calc(all_tau_aps_for_system_numbers_across_all_shuffles))[:6], size=8)

        elif plot_type == "maximum drop":
            od = None
            od = collections.OrderedDict(sorted(all_drops_for_system_numbers_across_all_shuffles.items()))
            df = pd.DataFrame.from_dict(od, orient='index')
            print df


            system_numbers_list = list(sorted(all_taus_for_system_numbers_across_all_shuffles.iterkeys()))
            data_frame = []
            for group_no, all_group_infos in sorted(all_drops_for_system_numbers_across_all_shuffles.iteritems()):
                tmp_grp_list = []
                for info in all_group_infos:
                    if np.isnan(info) == False:
                        tmp_grp_list.append(info)
                data_frame.append(tmp_grp_list)
            if datasource == 'TREC8' or datasource == 'TREC7':
                data_frame, system_numbers_list = update_dataframe(data_frame, datasource)

            plt.boxplot(data_frame)


            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
            #plt.boxplot(df)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > ((len(data_set_list) - 1) * len(plot_type_list)):
                plt.xlabel("number of groups")
            plt.ylabel("Max Drop")
            plt.ylim([0, 15])
            plt.yticks(np.arange(0, 15, step=2), size=8)

            plt.xticks(np.arange(1, len(system_numbers_list) + 1), system_numbers_list , size = 6)
            plt.grid(linestyle='dotted')
            plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(
                pearson_correlation_calc(all_drops_for_system_numbers_across_all_shuffles))[:6], size=8)

        elif plot_type == "unique number of documents":
            od = None
            od = collections.OrderedDict(sorted(all_unique_doc_counts_for_system_numbers_across_all_shuffles.items()))
            df = pd.DataFrame.from_dict(od, orient='index')
            print df
            system_numbers_list = list(sorted(all_taus_for_system_numbers_across_all_shuffles.iterkeys()))

            data_frame = []
            for group_no, all_group_infos in sorted(all_unique_doc_counts_for_system_numbers_across_all_shuffles.iteritems()):
                tmp_grp_list = []
                for info in all_group_infos:
                    if np.isnan(info) == False:
                        tmp_grp_list.append(info)
                data_frame.append(tmp_grp_list)

            if datasource == 'TREC8' or datasource == 'TREC7':
                data_frame, system_numbers_list = update_dataframe(data_frame, datasource)

            plt.boxplot(data_frame)
            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
            #plt.boxplot(df)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > ((len(data_set_list) - 1) * len(plot_type_list)):
                plt.xlabel("number of groups")
            plt.ylabel("# Unique Rel. Docs")
            plt.ylim([1000, 6500])
            plt.yticks(np.arange(0, 6500, step=1000), size=8)
            plt.xticks(np.arange(1, len(system_numbers_list) + 1), system_numbers_list, size = 6)

            plt.grid(linestyle='dotted')
            plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(
                pearson_correlation_calc(all_unique_doc_counts_for_system_numbers_across_all_shuffles))[:6], size=8)

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
plt.savefig(plot_address + 'ICTIR_figure_2_all_' + rankMetric +'.pdf', format='pdf', orientation='landscape')


