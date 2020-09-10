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
from numpy import trapz
from scipy.stats import pearsonr

from global_definition import *




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
#system_name_list = ['user-Indri-BM.uqv.run']
#document_selection_list = ['MABNS', 'MABNS2', 'MTF', 'RR', 'MAB']
document_selection_list = ['MAB_HIL', 'DYN_MTF_HIL']
data_set_list = ['WT2014', 'WT2013']
plot_type_list = ['tau', 'maximum drop', 'unique number of documents']
rankMetric = "map"
data_path = "/work/04549/mustaf/lonestar/data/collection/ClueWeb12UQV/"
query_type_list = ['diversity', 'popularity']
#query_type_list = ['diversity']


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(system_name_list), figsize=(15,8))
fig.subplots_adjust(hspace=0.9)

protocol_result = {}
plot_location = 1
onlyautomatic = 0

#linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted', 'densely dotted']
#linestyle_list = ['-g', '--c', '-.k', ':r', '--b']
linestyle_list = ['-g', '--c', '-.k', '^b', ':r']

for datasource in data_set_list:
    for systemName in system_name_list:
        plt.subplot(len(data_set_list), len(system_name_list), plot_location)
        recall_infos = {}
        index_for_linestyle = 0
        pool_depth_list = []
        pool_depth_list_names = []
        for document_selection_type in document_selection_list:
        # all_info_list = (tau_list, drop_list, relevant_count_list, recall_ratio_list, qrelsize_list)
            all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "all_info_list.pickle"
            all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
            recall_infos[document_selection_type] = all_info_list[3]


            list_of_budget = all_info_list[5]
            pool_depth_list = list_of_budget

            auc = trapz(recall_infos[document_selection_type], dx=10)

            plt.plot(pool_depth_list, recall_infos[document_selection_type], linestyle_list[index_for_linestyle], label = document_selection_type + '; AUC:' + str(auc)[:4], linewidth=2.0)
            index_for_linestyle = index_for_linestyle + 1

        #plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
        #for pool_depth_val in pool_depth_list:
        #    pool_depth_list_names.append("Pool " + str(pool_depth_val))

        plt.xticks(pool_depth_list, pool_depth_list, fontsize = 5, rotation=45)
        plt.grid(linestyle='dotted')
        if plot_location == 1 or plot_location == 6:
            plt.ylabel(datasource + "\nRecall")

        plt.ylim([0, 1])
        #plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)


        #plt.xlabel('% of human judgments', size=16)
        #plt.title(data_set_name_list[data_set_name_index], size=8)
        plt.grid(linestyle='dotted')
        #plt.xticks(x_labels_set, x_labels_set)
        plt.legend(loc='best', fontsize=6)
        if plot_location >= 6:
            plt.xlabel("Budget")
        plt.title(systemName, fontsize= 10)
        plot_location = plot_location + 1

plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'uqv_bandit_recall_budget.pdf', format='pdf', orientation='landscape')


plt.close()
plt.clf()

fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(system_name_list), figsize=(15,8))
fig.subplots_adjust(hspace=0.9)

plot_location = 1
for datasource in data_set_list:
    for systemName in system_name_list:
        plt.subplot(len(data_set_list), len(system_name_list), plot_location)
        recall_infos = {}
        index_for_linestyle = 0
        pool_depth_list = []
        pool_depth_list_names = []
        for document_selection_type in document_selection_list:
        # all_info_list = (tau_list, drop_list, relevant_count_list, recall_ratio_list, qrelsize_list)
            all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "all_info_list.pickle"
            all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
            recall_infos[document_selection_type] = all_info_list[0] # 0th index is for taus

            list_of_budget = all_info_list[5]
            pool_depth_list = list_of_budget

            auc = trapz(recall_infos[document_selection_type], dx=10)

            plt.plot(pool_depth_list, recall_infos[document_selection_type], linestyle_list[index_for_linestyle], label = document_selection_type + '; AUC:' + str(auc)[:4], linewidth=2.0)
            index_for_linestyle = index_for_linestyle + 1

        #plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
        plt.xticks(pool_depth_list, pool_depth_list, fontsize = 5, rotation=45)
        plt.grid(linestyle='dotted')
        if plot_location == 1 or plot_location == 6:
            plt.ylabel(datasource + "\ntau correlation")

        plt.ylim([0.4, 1])
        #plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)


        #plt.xlabel('% of human judgments', size=16)
        #plt.title(data_set_name_list[data_set_name_index], size=8)
        plt.grid(linestyle='dotted')
        #plt.xticks(x_labels_set, x_labels_set)
        plt.legend(loc='best', fontsize=6)
        if plot_location >= 6:
            plt.xlabel("Budget")
        plt.title(systemName, fontsize= 10)
        plot_location = plot_location + 1

plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'uqv_bandit_taus_budget.pdf', format='pdf', orientation='landscape')



plt.close()
plt.clf()

fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(system_name_list), figsize=(15,8))
fig.subplots_adjust(hspace=0.9)

plot_location = 1
for datasource in data_set_list:
    for systemName in system_name_list:
        plt.subplot(len(data_set_list), len(system_name_list), plot_location)
        recall_infos = {}
        index_for_linestyle = 0
        pool_depth_list = []
        pool_depth_list_names = []
        for document_selection_type in document_selection_list:
        # all_info_list = (tau_list, drop_list, relevant_count_list, recall_ratio_list, qrelsize_list)
            all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_" + document_selection_type + "all_info_list.pickle"
            all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
            recall_infos[document_selection_type] = all_info_list[1] # 0th index is for taus

            list_of_budget = all_info_list[5]
            pool_depth_list = list_of_budget

            auc = trapz(recall_infos[document_selection_type], dx=10)

            plt.plot(pool_depth_list, recall_infos[document_selection_type], linestyle_list[index_for_linestyle], label = document_selection_type + '; AUC:' + str(auc)[:6], linewidth=2.0)
            index_for_linestyle = index_for_linestyle + 1

        #plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)

        plt.xticks(pool_depth_list, pool_depth_list, fontsize = 5, rotation=45)
        plt.grid(linestyle='dotted')
        if plot_location == 1 or plot_location == 6:
            plt.ylabel(datasource + "\nMax Drop")

        plt.ylim([0, 25])
        #plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)


        #plt.xlabel('% of human judgments', size=16)
        #plt.title(data_set_name_list[data_set_name_index], size=8)
        plt.grid(linestyle='dotted')
        #plt.xticks(x_labels_set, x_labels_set)
        plt.legend(loc='best', fontsize=6)
        if plot_location >= 6:
            plt.xlabel("Budget")
        plt.title(systemName, fontsize= 10)
        plot_location = plot_location + 1

plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'uqv_bandit_maxdrop_budget.pdf', format='pdf', orientation='landscape')




exit(0)

## Plot # 2
plt.close()

plt.clf()
import os
from statistics import mean


fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(query_type_list), figsize=(5,4))
fig.subplots_adjust(hspace=0.9)


plot_location = 1


data_set_list = ['WT2013', 'WT2014']
for datasource in data_set_list:
    plt.subplot(1, len(data_set_list), plot_location)

    TopicDiversityInfoSorted_filename = data_path + datasource + "_" + systemName + "_" + "diverse_query_by_tfidfscore.pickle"
    print "topic_query_diversity_info_file:", TopicDiversityInfoSorted_filename
    TopicDiversityInfoSorted = pickle.load(open(TopicDiversityInfoSorted_filename, "rb"))


    TopicPopularQuertFilename = data_path + datasource + "_" + systemName + "_popular_query_by_relevant_docs" + '.pickle'
    print "Popular Query by Relevant Doc:", TopicPopularQuertFilename
    topicQueryInfo = pickle.load(open(TopicPopularQuertFilename, "rb"))


    topic_list = list(sorted(TopicDiversityInfoSorted.keys()))
    print topic_list
    topic_top5_mean_diversityscores_list = []
    topic_bottom5_mean_diversityscores_list = []

    for topicNo, queryList in sorted(topicQueryInfo.iteritems()):
        #print topicNo
        top5popular_queries = queryList[0:5]
        bottom5popular_queties = queryList[-5:]

        top5popular_queries_diversityScores = []
        bottom5popular_queties_diversityScores = []

        for query in top5popular_queries:
            top5popular_queries_diversityScores.append(TopicDiversityInfoSorted[topicNo][query])

        for query in bottom5popular_queties:
            bottom5popular_queties_diversityScores.append(TopicDiversityInfoSorted[topicNo][query])

        topic_top5_mean_diversityscores_list.append(mean(top5popular_queries_diversityScores))
        topic_bottom5_mean_diversityscores_list.append(mean(bottom5popular_queties_diversityScores))

    plt.plot(topic_list, topic_top5_mean_diversityscores_list, linestyle_list[0],
                 label= "top 5 query variants by productivity")

    plt.plot(topic_list, topic_bottom5_mean_diversityscores_list, linestyle_list[1],
                 label= "bottom 5 query variants by productivity")

    plt.xticks(topic_list, topic_list, fontsize=5, rotation=90)
    plt.xlabel("topic Id")
    if plot_location == 1:
        plt.ylabel("mean average similarity score")
    plt.ylim([0, 1])

    plt.grid(linestyle='dotted')

    plt.legend(loc='best', fontsize=6)
    plt.title(datasource)

    plot_location = plot_location + 1


plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'uqvtfidfdiversity.pdf', format='pdf', orientation='landscape')

### Table 1

all_infos_by_datasource = {} # key is the datasource
for datasource in data_set_list:
    all_info_by_query_type = {}
    for query_type in query_type_list:
        plt.subplot(len(data_set_list), len(query_type_list), plot_location)

        all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + query_type + "_all_info_list.pickle"
        all_info_list = pickle.load(open(all_info_list_file_name, "rb"))

        all_info_by_query_type[query_type] = all_info_list

    all_infos_by_datasource[datasource] = all_info_by_query_type

def round_by_4_places(num):
    return str(num)[0:6]

for i in xrange(0, 50):
    print i+1, "&",round_by_4_places(all_infos_by_datasource['WT2013']['diversity'][0][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2013']['popularity'][0][i]), "&", \
        round_by_4_places(all_infos_by_datasource['WT2013']['diversity'][1][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2013']['popularity'][1][i]), "&", \
        round_by_4_places(all_infos_by_datasource['WT2014']['diversity'][0][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['popularity'][0][i]), "&", \
        round_by_4_places(all_infos_by_datasource['WT2014']['diversity'][1][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['popularity'][1][i]), "\\\\"




## Table 2 --> qrels size
for i in xrange(0, 50):
    print i+1, "&",round_by_4_places(all_infos_by_datasource['WT2013']['diversity'][3][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2013']['popularity'][3][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['diversity'][3][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['popularity'][3][i]),"\\\\"



## Table 3 --> per system wise analysis
system_name_list = ['user-Atire-MC2.uqv.run','user-Indri-BM.uqv.run', 'user-Indri-LM.uqv.run', 'user-Terrier-DFR.uqv.run', 'user-Terrier-PLC.uqv.run']

for systemName in system_name_list:
    all_infos_by_datasource = {}  # key is the datasource
    for datasource in data_set_list:
        all_info_by_query_type = {}
        for query_type in query_type_list:
            plt.subplot(len(data_set_list), len(query_type_list), plot_location)

            all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + query_type + "_all_info_list.pickle"
            all_info_list = pickle.load(open(all_info_list_file_name, "rb"))

            all_info_by_query_type[query_type] = all_info_list

        all_infos_by_datasource[datasource] = all_info_by_query_type

    i = 14 # 15th query variant we are looking for
    print systemName, "&", round_by_4_places(all_infos_by_datasource['WT2013']['diversity'][0][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2013']['popularity'][0][i]), "&", \
        round_by_4_places(all_infos_by_datasource['WT2013']['diversity'][1][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2013']['popularity'][1][i]), "&", \
        round_by_4_places(all_infos_by_datasource['WT2013']['diversity'][2][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2013']['popularity'][2][i]), "&", \
        round_by_4_places(all_infos_by_datasource['WT2014']['diversity'][0][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['popularity'][0][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['diversity'][1][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['popularity'][1][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['diversity'][2][i]), \
        "&", round_by_4_places(all_infos_by_datasource['WT2014']['popularity'][2][i]), "\\\\"
