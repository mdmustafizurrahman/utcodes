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
#system_name_list = ['user-Atire-MC2.uqv.run','user-Indri-BM.uqv.run', 'user-Indri-LM.uqv.run', 'user-Terrier-DFR.uqv.run', 'user-Terrier-PLC.uqv.run']
system_name_list = ['user-Indri-BM.uqv.run']
data_set_list = ['WT2013', 'WT2014']
plot_type_list = ['tau', 'maximum drop', 'unique number of documents']
datasource = data_set_list[0]
systemName = system_name_list[0]
rankMetric = "map"
data_path = "/work/04549/mustaf/lonestar/data/collection/ClueWeb12UQV/"
query_type_list = ['diversity', 'popularity']
#query_type_list = ['diversity']


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(query_type_list), figsize=(10,8))
fig.subplots_adjust(hspace=0.9)

protocol_result = {}
plot_location = 1
onlyautomatic = 0

#linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted', 'densely dotted']
#linestyle_list = ['-g', '--c', '-.k', ':r', '--b']
linestyle_list = ['-g', '--c', '-.k', '--b']

for datasource in data_set_list:
    for query_type in query_type_list:
        plt.subplot(len(data_set_list), len(query_type_list), plot_location)

        all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + query_type + "_all_info_list.pickle"

        all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
        rmse_list = all_info_list[2]

        query_variant_list = [0, 4, 9, 14]
        for index_for_linestyle, queryVariantsNumber in enumerate(query_variant_list):

            pseudo_qrels_topic_relevant_counts_filename = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + query_type + "_" + str(
                queryVariantsNumber) + "_topic_relevantDocCounts.pickle"

            topic_relevants = pickle.load(open(pseudo_qrels_topic_relevant_counts_filename, "rb"))
            tmp_list = []
            for topicNo, releCount in sorted(topic_relevants.iteritems()):
                tmp_list.append(releCount)
                print topicNo, releCount

            topic_list = list(sorted(topic_relevants.keys()))
            rmse_val = rmse_list[queryVariantsNumber]

            plt.plot(topic_list, tmp_list, linestyle_list[index_for_linestyle], label = str(queryVariantsNumber + 1) + " query variants; "+ "RMSE="+ str(rmse_val)[0:5])


        #plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
        plt.xticks(topic_list, topic_list, fontsize = 5, rotation=90)
        plt.grid(linestyle='dotted')
        plt.ylabel("# of Rel. Docs")
        plt.ylim([0, 300])
        #plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)

        plot_location = plot_location + 1

        #plt.xlabel('% of human judgments', size=16)
        #plt.title(data_set_name_list[data_set_name_index], size=8)
        plt.grid(linestyle='dotted')
        #plt.xticks(x_labels_set, x_labels_set)
        plt.legend(loc='best', fontsize=6)
        plt.xlabel("topic Id")
        query_selection_str = query_type
        if query_type == 'popularity':
            query_selection_str = 'productivity'
        plt.title(datasource + "\nquery selection by " + query_selection_str, fontsize= 10)

plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'uqvrmse.pdf', format='pdf', orientation='landscape')


## Plot # 2
plt.close()

plt.clf()
import os
from statistics import mean


fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(query_type_list), figsize=(10,4))
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
