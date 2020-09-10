import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)
import math
import pickle
import sys
from global_definition import *
from tqdm import tqdm
import os
from multiprocessing import Pool as ProcessPool
import itertools
from functools import partial
import numpy as np

def topic_relevant_percentage(topicID, topic_complete_qrels_address):
    topic_complete_qrels_file_name = topic_complete_qrels_address + str(topicID) + '.pickle'
    topic_complete_qrels = pickle.load(open(topic_complete_qrels_file_name, 'rb'))

    original_labels = topic_complete_qrels[0]
    predicted_label = topic_complete_qrels[1]

    original_labels_list = []
    for k, v in original_labels.iteritems():
        original_labels_list.append(v)

    predicted_labels_list = []
    for k, v in predicted_label.iteritems():
        predicted_labels_list.append(v)

    original_qrels_number_relevants = original_labels_list.count(1)
    predicted_qrels_number_relevants = predicted_labels_list.count(1)

    original_per_centage = ((original_qrels_number_relevants * 1.0) / (
                original_qrels_number_relevants + predicted_qrels_number_relevants)) * 100
    predicted_per_centage = 100.0 - original_per_centage

    return (topicID, original_per_centage, predicted_per_centage)

def topic_relevant_percentage_multiprocessing(topic_list, topic_complete_qrels_file_name):
    topic_original_relevant_docs_ratio = {}  # keep the percentage of relevant from original
    topic_predicted_relevant_docs_ratio = {}  # keep the percentage of relevant from predcited
    num_workers = None
    workers = ProcessPool(num_workers)

    with tqdm(total=len(topic_list)) as pbar:
        partial_topic_relevant_percentage = partial(topic_relevant_percentage, topic_complete_qrels_address=topic_complete_qrels_address)  # prod_x has only one argument x (y is fixed to 10)
        for results in tqdm(workers.imap_unordered(partial_topic_relevant_percentage, topic_list)):
            original_per_centage = results[1]
            predicted_per_centage = results[2]
            topic_id = results[0]
            topic_original_relevant_docs_ratio[topic_id] = original_per_centage
            topic_predicted_relevant_docs_ratio[topic_id] = predicted_per_centage
            pbar.update()
    return (topic_original_relevant_docs_ratio, topic_predicted_relevant_docs_ratio)


if __name__ == '__main__':

    datasource = sys.argv[1]


    bar_colors_list = ["#006D2C", "#74C476"]
    data_path = base_address + datasource + "/result/"
    topic_complete_qrels_address = data_path + "per_topic_complete_qrels_" + classifier_name + "_"
    plot_address = base_address + 'plot/'

    topic_list = [str(topicID) for topicID in xrange(start_topic[datasource], end_topic[datasource])]
    original_dict, predicted_dict = topic_relevant_percentage_multiprocessing(topic_list,topic_complete_qrels_address)
    originial_percentage_list = []
    predicted_percentage_list = []

    for topicID in sorted(topic_list):
        #print topicID
        originial_percentage_list.append(original_dict[topicID])
        predicted_percentage_list.append(predicted_dict[topicID])

    plt.bar(topic_list, originial_percentage_list, color = bar_colors_list[0])
    plt.bar(topic_list, predicted_percentage_list, color = bar_colors_list[1], bottom = originial_percentage_list)
    plt.xticks(np.arange(0, len(topic_list),1), topic_list, rotation=70, size = 8)
    #plt.legend(loc=1)
    plt.xlabel("Topic Id")
    plt.ylabel("Percentage of relevant documents")
    plt.title(datasource)

    plt.savefig(plot_address + 'stacked_bar_'+ datasource +'.pdf', format='pdf')
