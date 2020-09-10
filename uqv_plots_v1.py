import matplotlib
matplotlib.use('Agg')
import seaborn as sns; sns.set()
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
import numpy as np
import os
from statistics import mean

import pandas as pd
from numpy import trapz

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

data_set_list = ['WT2013', 'WT2014']
plot_type_list = ['tau', 'maxdrop', 'recall', 'budget']
rankMetric = "map"
data_path = "/work/04549/mustaf/lonestar/data/collection/ClueWeb12UQV/"
#ordering_of_query_list = ['most_diverse', 'most_popular', 'least_diverse', 'least_popular', 'random_sample', 'soumya_most_popular', 'alex_most_popular']
ordering_of_query_list = ['most_diverse', 'random_sample', 'alex_most_popular']

#ordering_of_query_list = ['most_diverse', 'most_popular', 'least_diverse', 'least_popular']

#query_type_list = ['diversity']


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(plot_type_list), figsize=(14,8))
fig.subplots_adjust(hspace=0.9)

protocol_result = {}
plot_location = 1
onlyautomatic = 0

#linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted', 'densely dotted']
#linestyle_list = ['-g', '--c', '-.k', ':r', '--b']
linestyle_list = ['-g', '*-c', '-.b', '-^k', ':r', '-oy',  '.--m']

numberOfSamples = 50
systemName = system_name_list[0]
import numpy
def numpy_mean_calculation(x):
    #print x[1]
    #print x[49]
    #print len(x)
    y = np.array([[list_item for list_item in listofvalues] for sample_number, listofvalues in x.items()])

    #print y
    return np.mean(y, axis = 0)
    #exit(0)
    #return np.mean([np.array(x[1]) ,np.array(x[2]) ,np.array(x[3]),np.array(x[4])], axis= 0).tolist()

def mean_samples(sample_info):

    tau_info = {}
    drop_info = {}
    rmse_info = {}
    recall_info = {}
    qrel_info = {}

    for sample_number, sample_all_info in sorted(sample_info.iteritems()):
        tau_info[sample_number] =  sample_all_info[0]
        drop_info[sample_number] =  sample_all_info[1]
        rmse_info[sample_number] =  sample_all_info[2]
        qrel_info[sample_number] = sample_all_info[3]
        recall_info[sample_number] =  sample_all_info[4]

    tau_all = numpy_mean_calculation(tau_info)
    drop_all = numpy_mean_calculation(drop_info)
    rmse_all = numpy_mean_calculation(rmse_info)
    qrel_all = numpy_mean_calculation(qrel_info)
    recall_all = numpy_mean_calculation(recall_info)

    return (tau_all, drop_all, rmse_all, qrel_all, recall_all)

numOfVariantstoPlot = 50
for datasource in data_set_list:
    all_infos = {}

    for ordering_of_query in ordering_of_query_list:
        if ordering_of_query != "random_sample":
            all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_all_info_list.pickle"
            all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
            # format --> all_info_list = (tau_list, drop_list, rmse_list, qrelsize_list, recall_list)
            all_infos[ordering_of_query] =  all_info_list
        elif ordering_of_query == 'random_sample':
            sample_info = {}
            for sample_number in xrange(1, numberOfSamples + 1):
                all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_" + str(
                    sample_number) + "_all_info_list.pickle"
                all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
                sample_info[sample_number] = all_info_list
            all_infos[ordering_of_query] =mean_samples(sample_info)
    for plot_type in plot_type_list:
        plt.subplot(len(data_set_list), len(plot_type_list), plot_location)

        if plot_type == 'tau':
            for index_for_linestyle, ordering_of_query in enumerate(ordering_of_query_list):
                tau_list = all_infos[ordering_of_query][0][0:numOfVariantstoPlot]
                x_labels = list(xrange(1,numOfVariantstoPlot + 1,1))

                auc = trapz(tau_list, dx=10)

                plt.plot(x_labels, tau_list, linestyle_list[index_for_linestyle], label = ordering_of_query + '; AUC:' + str(auc)[:8])


                #plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
                plt.xticks(x_labels, x_labels, fontsize = 3, rotation=90)
                plt.grid(linestyle='dotted')
                plt.ylabel("tau")
                plt.ylim([0.8, 1])
                #plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)


                #plt.xlabel('% of human judgments', size=16)
                #plt.title(data_set_name_list[data_set_name_index], size=8)
                plt.grid(linestyle='dotted')
                #plt.xticks(x_labels_set, x_labels_set)
                plt.legend(loc='best', fontsize=6)
                plt.xlabel("# of query variants")
                plt.title(datasource, fontsize= 10)
            plot_location = plot_location + 1

        elif plot_type == 'maxdrop':
            for index_for_linestyle, ordering_of_query in enumerate(ordering_of_query_list):
                tau_list = all_infos[ordering_of_query][1][0:numOfVariantstoPlot]
                x_labels = list(xrange(1, numOfVariantstoPlot + 1, 1))

                auc = trapz(tau_list, dx=10)

                plt.plot(x_labels, tau_list, linestyle_list[index_for_linestyle], label=ordering_of_query + '; AUC:' + str(auc)[:8])

                # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
                plt.xticks(x_labels, x_labels, fontsize=3, rotation=90)
                plt.grid(linestyle='dotted')
                plt.ylabel("Max Drop")
                plt.ylim([0, 10])
                # plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)


                # plt.xlabel('% of human judgments', size=16)
                # plt.title(data_set_name_list[data_set_name_index], size=8)
                plt.grid(linestyle='dotted')
                # plt.xticks(x_labels_set, x_labels_set)
                plt.legend(loc='best', fontsize=6)
                plt.xlabel("# of query variants")
                plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1

        elif plot_type == 'recall':
            for index_for_linestyle, ordering_of_query in enumerate(ordering_of_query_list):
                tau_list = all_infos[ordering_of_query][4][0:numOfVariantstoPlot]
                x_labels = list(xrange(1, numOfVariantstoPlot+1, 1))

                auc = trapz(tau_list, dx=10)

                plt.plot(x_labels, tau_list, linestyle_list[index_for_linestyle], label=ordering_of_query + '; AUC:' + str(auc)[:8])

                # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
                plt.xticks(x_labels, x_labels, fontsize=3, rotation=90)
                plt.grid(linestyle='dotted')
                plt.ylabel("Recall")
                plt.ylim([0, 1])
                # plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)


                # plt.xlabel('% of human judgments', size=16)
                # plt.title(data_set_name_list[data_set_name_index], size=8)
                plt.grid(linestyle='dotted')
                # plt.xticks(x_labels_set, x_labels_set)
                plt.legend(loc='best', fontsize=6)
                plt.xlabel("# of query variants")
                plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1

        elif plot_type == 'budget':
            for index_for_linestyle, ordering_of_query in enumerate(ordering_of_query_list):
                tau_list = all_infos[ordering_of_query][3][0:numOfVariantstoPlot]
                x_labels = list(xrange(1, numOfVariantstoPlot + 1, 1))

                auc = trapz(tau_list, dx=10)

                plt.plot(x_labels, tau_list, linestyle_list[index_for_linestyle], label=ordering_of_query + '; AUC:' + str(auc)[:8])

                # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
                plt.xticks(x_labels, x_labels, fontsize=3, rotation=90)
                plt.grid(linestyle='dotted')
                plt.ylabel("Total Judged")
                plt.ylim([1000, 22000])
                # plt.yticks(np.arange(0.0, 1.01, step=0.1), size=8)


                # plt.xlabel('% of human judgments', size=16)
                # plt.title(data_set_name_list[data_set_name_index], size=8)
                plt.grid(linestyle='dotted')
                # plt.xticks(x_labels_set, x_labels_set)
                plt.legend(loc='best', fontsize=6)
                plt.xlabel("# of query variants")
                plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1

plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'uqv_plot_1_50.pdf', format='pdf', orientation='landscape')



## Plot # 2
plt.close()
plt.clf()



fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(plot_type_list), figsize=(14,8))
fig.subplots_adjust(hspace=0.9)

plot_location = 1

ordering_of_query = 'most_diverse'
for datasource in data_set_list:
    all_infos = {}

    for systemName in system_name_list:
        all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_all_info_list.pickle"
        all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
        # format --> all_info_list = (tau_list, drop_list, rmse_list, qrelsize_list, recall_list)
        all_infos[systemName] = all_info_list

    for plot_type in plot_type_list:
        plt.subplot(len(data_set_list), len(plot_type_list), plot_location)

        if plot_type == 'tau':

            y_list = []
            x_list = []
            for queryVariantNo in xrange(0,numOfVariantstoPlot):
                for systemName in system_name_list:
                    y_list.append(all_infos[systemName][0][queryVariantNo])
                tmp_x_list = [queryVariantNo+1]*len(system_name_list)
                x_list = x_list + tmp_x_list

            d = {'X': x_list, 'Y': y_list}
            df = pd.DataFrame(data=d)
            sns.lineplot(x="X", y="Y", data=df, label=ordering_of_query)

            # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
            plt.xticks(x_labels, x_labels, fontsize=3, rotation=90)
            plt.grid(linestyle='dotted')
            plt.ylabel("tau")
            plt.ylim([0.6, 1])

            plt.grid(linestyle='dotted')
            #plt.legend(loc='best', fontsize=6)
            plt.xlabel("# of query variants")
            plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1

        elif plot_type == 'maxdrop':

            y_list = []
            x_list = []
            for queryVariantNo in xrange(0,numOfVariantstoPlot):
                for systemName in system_name_list:
                    y_list.append(all_infos[systemName][1][queryVariantNo])
                tmp_x_list = [queryVariantNo+1]*len(system_name_list)
                x_list = x_list + tmp_x_list

            d = {'X': x_list, 'Y': y_list}
            df = pd.DataFrame(data=d)
            sns.lineplot(x="X", y="Y", data=df, label=ordering_of_query)

            # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
            plt.xticks(x_labels, x_labels, fontsize=8, rotation=90)
            plt.grid(linestyle='dotted')
            plt.ylabel("Max Drop")
            plt.ylim([0, 25])

            plt.grid(linestyle='dotted')
            #plt.legend(loc='best', fontsize=6)
            plt.xlabel("# of query variants")
            plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1

        elif plot_type == 'recall':

            y_list = []
            x_list = []
            for queryVariantNo in xrange(0,numOfVariantstoPlot):
                for systemName in system_name_list:
                    y_list.append(all_infos[systemName][4][queryVariantNo])
                tmp_x_list = [queryVariantNo+1]*len(system_name_list)
                x_list = x_list + tmp_x_list

            d = {'X': x_list, 'Y': y_list}
            df = pd.DataFrame(data=d)
            sns.lineplot(x="X", y="Y", data=df, label=ordering_of_query)

            # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
            plt.xticks(x_labels, x_labels, fontsize=8, rotation=90)
            plt.grid(linestyle='dotted')
            plt.ylabel("Recall")
            plt.ylim([0, 1])

            plt.grid(linestyle='dotted')
            #plt.legend(loc='best', fontsize=6)
            plt.xlabel("# of query variants")
            plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1

        elif plot_type == 'budget':

            y_list = []
            x_list = []
            for queryVariantNo in xrange(0,numOfVariantstoPlot):
                for systemName in system_name_list:
                    y_list.append(all_infos[systemName][3][queryVariantNo])
                tmp_x_list = [queryVariantNo+1]*len(system_name_list)
                x_list = x_list + tmp_x_list

            d = {'X': x_list, 'Y': y_list}
            df = pd.DataFrame(data=d)
            sns.lineplot(x="X", y="Y", data=df, label=ordering_of_query)

            # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
            plt.xticks(x_labels, x_labels, fontsize=8, rotation=90)
            plt.grid(linestyle='dotted')
            plt.ylabel("Total Judged")
            plt.ylim([1000, 22000])

            plt.grid(linestyle='dotted')
            #plt.legend(loc='best', fontsize=6)
            plt.xlabel("# of query variants")
            plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1



plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'uqv_plot_2_50.pdf', format='pdf', orientation='landscape')



plt.close()
plt.clf()



fig, ax = plt.subplots(nrows=len(data_set_list), ncols=len(plot_type_list), figsize=(14,8))
fig.subplots_adjust(hspace=0.9)

plot_location = 1
queryVariantNo = 15 - 1

for datasource in data_set_list:
    all_infos = {}

    for ordering_of_query in ordering_of_query_list:
        all_info_ordering = {}
        for systemName in system_name_list:
            if ordering_of_query == 'random_sample':
                sample_info = {}
                for sample_number in xrange(1, numberOfSamples + 1):
                    all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_" + str(
                        sample_number) + "_all_info_list.pickle"
                    all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
                    sample_info[sample_number] = all_info_list
                all_info_ordering[systemName] = mean_samples(sample_info)
            else:
                all_info_list_file_name = data_path + "qrels_" + datasource + "_" + systemName + "_query_variants_by_" + ordering_of_query + "_all_info_list.pickle"
                all_info_list = pickle.load(open(all_info_list_file_name, "rb"))
                # format --> all_info_list = (tau_list, drop_list, rmse_list, qrelsize_list, recall_list)
                all_info_ordering[systemName] = all_info_list
        all_infos[ordering_of_query] = all_info_ordering

    for plot_type in plot_type_list:
        plt.subplot(len(data_set_list), len(plot_type_list), plot_location)

        if plot_type == 'tau':

            y_list = []
            x_list = []
            for ordering_index, ordering_of_query in enumerate(ordering_of_query_list):
                for systemName in system_name_list:
                    y_list.append(all_infos[ordering_of_query][systemName][0][queryVariantNo])
                tmp_x_list = [ordering_index+1]*len(system_name_list)
                x_list = x_list + tmp_x_list

            d = {'X': x_list, 'Y': y_list}
            df = pd.DataFrame(data=d)
            sns.boxplot(x="X", y="Y", data=df)

            # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
            plt.xticks(list(xrange(0, len(ordering_of_query_list))), ordering_of_query_list, fontsize=8, rotation=90)
            plt.grid(linestyle='dotted')
            plt.ylabel("tau")
            plt.ylim([0.8, 1])

            plt.grid(linestyle='dotted')
            #plt.legend(loc='best', fontsize=6)
            plt.xlabel("query selection type")
            plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1


        elif plot_type == 'maxdrop':

            y_list = []
            x_list = []
            for ordering_index, ordering_of_query in enumerate(ordering_of_query_list):
                for systemName in system_name_list:
                    y_list.append(all_infos[ordering_of_query][systemName][1][queryVariantNo])
                tmp_x_list = [ordering_index+1]*len(system_name_list)
                x_list = x_list + tmp_x_list

            d = {'X': x_list, 'Y': y_list}
            df = pd.DataFrame(data=d)
            sns.boxplot(x="X", y="Y", data=df)

            # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
            plt.xticks(list(xrange(0, len(ordering_of_query_list))), ordering_of_query_list, fontsize=8, rotation=90)
            plt.grid(linestyle='dotted')
            plt.ylabel("Max Drop")
            plt.ylim([0, 10])

            plt.grid(linestyle='dotted')
            #plt.legend(loc='best', fontsize=6)
            plt.xlabel("query selection type")
            plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1

        elif plot_type == 'recall':

            y_list = []
            x_list = []
            for ordering_index, ordering_of_query in enumerate(ordering_of_query_list):
                for systemName in system_name_list:
                    y_list.append(all_infos[ordering_of_query][systemName][4][queryVariantNo])
                tmp_x_list = [ordering_index+1]*len(system_name_list)
                x_list = x_list + tmp_x_list

            d = {'X': x_list, 'Y': y_list}
            df = pd.DataFrame(data=d)
            sns.boxplot(x="X", y="Y", data=df)

            # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
            plt.xticks(list(xrange(0, len(ordering_of_query_list))), ordering_of_query_list, fontsize=8, rotation=90)
            plt.grid(linestyle='dotted')
            plt.ylabel("Recall")
            plt.ylim([0.4, 1])

            plt.grid(linestyle='dotted')
            #plt.legend(loc='best', fontsize=6)
            plt.xlabel("query selection type")
            plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1
        elif plot_type == 'budget':

            y_list = []
            x_list = []
            for ordering_index, ordering_of_query in enumerate(ordering_of_query_list):
                for systemName in system_name_list:
                    y_list.append(all_infos[ordering_of_query][systemName][3][queryVariantNo])
                tmp_x_list = [ordering_index+1]*len(system_name_list)
                x_list = x_list + tmp_x_list

            d = {'X': x_list, 'Y': y_list}
            df = pd.DataFrame(data=d)
            sns.boxplot(x="X", y="Y", data=df)

            # plt.xticks(np.arange(1, len(topic_list) + 1), topic_list)
            plt.xticks(list(xrange(0, len(ordering_of_query_list))), ordering_of_query_list, fontsize=8, rotation=90)
            plt.grid(linestyle='dotted')
            plt.ylabel("Total Judged")
            plt.ylim([8000, 22000])

            plt.grid(linestyle='dotted')
            #plt.legend(loc='best', fontsize=6)
            plt.xlabel("query selection type")
            plt.title(datasource, fontsize=10)
            plot_location = plot_location + 1



plt.tight_layout()
print plot_address
plt.savefig(plot_address + 'uqv_plot_3_50.pdf', format='pdf', orientation='landscape')
