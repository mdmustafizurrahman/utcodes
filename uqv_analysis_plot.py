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






#data_set_list = ['TREC7','TREC8', 'gov2', 'WT2013', 'WT2014']
system_name_list = ['user-Atire-MC2.uqv.run','user-Indri-BM.uqv.run', 'user-Indri-LM.uqv.run', 'user-Terrier-DFR.uqv.run', 'user-Terrier-PLC.uqv.run']
data_set_list = ['WT2013']
plot_type_list = ['tau', 'maximum drop', 'unique number of documents']
datasource = data_set_list[0]
rankMetric = "map"
data_path = "/work/04549/mustaf/lonestar/data/collection/ClueWeb12UQV/"


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(system_name_list), ncols=len(plot_type_list), figsize=(10,12))
fig.subplots_adjust(hspace=0.9)

protocol_result = {}
plot_location = 1
onlyautomatic = 0

for system_name_index, systemName in enumerate(system_name_list):

    queryVariantsTauFileName = data_path + "qrels_" + datasource + "_" + systemName + "_queryVariants_Tau.pickle"
    queryVariantsDropFileName = data_path + "qrels_" + datasource + "_" + systemName + "_queryVariants_MaxDrop.pickle"
    queryVariantsUniqRelDictFileName = data_path + "qrels_" + datasource + "_" + systemName + "_queryVariants_Unique_Rel_Count.pickle"

    queryVariantsTauDict = pickle.load(open(queryVariantsTauFileName, "rb"))
    queryVariantsDropDict = pickle.load(open(queryVariantsDropFileName, "rb"))
    queryVariantsUniqRelDict = pickle.load(open(queryVariantsUniqRelDictFileName, "rb"))



    for plot_type in plot_type_list:
        # subplot starting
        plt.subplot(len(system_name_list), len(plot_type_list), plot_location)

        if plot_type == "tau":
            od = None
            od = collections.OrderedDict(sorted(queryVariantsTauDict.items()))
            df = pd.DataFrame.from_dict(od, orient='index')

            query_varaints_numbers_list = list(sorted(queryVariantsTauDict.iterkeys()))
            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):


            print df

            plt.boxplot(df)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > ((len(system_name_list)-1)*len(plot_type_list)):
                plt.xlabel("number of groups")
            plt.ylabel("tau")
            plt.ylim([0, 1])
            plt.yticks(np.arange(0.0, 1.01, step=0.1), size = 8)
            plt.xticks(np.arange(1, len(query_varaints_numbers_list) + 1), query_varaints_numbers_list)
            plt.grid(linestyle='dotted')
            #plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(pearson_correlation_calc(all_taus_for_system_numbers_across_all_shuffles))[:6], size=8)


        elif plot_type == "maximum drop":
            od = None
            od = collections.OrderedDict(sorted(queryVariantsDropDict.items()))
            df = pd.DataFrame.from_dict(od, orient='index')

            query_varaints_numbers_list = list(sorted(queryVariantsDropDict.iterkeys()))

            print df

            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
            plt.boxplot(df)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > ((len(system_name_list)-1)*len(plot_type_list)):

                plt.xlabel("number of groups")
            plt.ylabel("Max Drop")
            plt.ylim([0, 15])
            plt.yticks(np.arange(0, 15, step=2), size=8)

            plt.xticks(np.arange(1, len(query_varaints_numbers_list) + 1), query_varaints_numbers_list)
            plt.grid(linestyle='dotted')
            #plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(pearson_correlation_calc(all_drops_for_system_numbers_across_all_shuffles))[:6], size=8)

        elif plot_type == "unique number of documents":
            od = None
            od = collections.OrderedDict(sorted(queryVariantsUniqRelDict.items()))
            df = pd.DataFrame.from_dict(od, orient='index')

            query_varaints_numbers_list = list(sorted(queryVariantsUniqRelDict.iterkeys()))

            print df

            # for system_numbers, all_taus_for_system_numbers in sorted(all_taus_for_system_numbers_across_all_shuffles.iteritems()):
            plt.boxplot(df)
            # plt.xticks(x_labels, x_labels_set)
            if plot_location > ((len(system_name_list)-1)*len(plot_type_list)):
                plt.xlabel("number of groups")
            plt.ylabel("# Unique Rel. Docs")
            plt.ylim([1000, 6000])
            plt.yticks(np.arange(0, 6000, step=1000), size=8)
            plt.xticks(np.arange(1, len(query_varaints_numbers_list) + 1), query_varaints_numbers_list)

            plt.grid(linestyle='dotted')
            #plt.title(data_set_name_list[data_set_name_index] + "\n pearson correlation = " + str(pearson_correlation_calc(all_unique_doc_counts_for_system_numbers_across_all_shuffles))[:6], size=8)

        plot_location = plot_location + 1

    #plt.xlabel('% of human judgments', size=16)
    #plt.title(data_set_name_list[data_set_name_index], size=8)
    plt.grid(linestyle='dotted')
    #plt.xticks(x_labels_set, x_labels_set)


plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'UQV_'+ data_set_list[0] +"_" + rankMetric +'.pdf', format='pdf', orientation='landscape')


