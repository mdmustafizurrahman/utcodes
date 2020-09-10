import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)
import math
import pickle
import sys
import operator
from global_definition import *

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

plot_type = sys.argv[1] # can be 'tau' or 'f1'

classifier_name_list = ['NR']
data_set_list = ['TREC7','TREC8']
al_protocol_list = ['CAL', 'SAL', 'SPL']
rankMetric = 'infAP'
seed_selection_type = 'IS'


plot_address = base_address + 'plot/'
fig, ax = plt.subplots(nrows=len(classifier_name_list), ncols=len(data_set_list), figsize=(10,3))

protocol_result = {}
plot_location = 1
for classifier_name in classifier_name_list:
    for data_set_name_index, datasource in enumerate(data_set_list):
        data_path = base_address + datasource + "/result/"
        topic_list = [str(topicID) for topicID in xrange(start_topic[datasource], end_topic[datasource])]
        number_of_topic = len(topic_list)

        # subplot starting
        plt.subplot(len(classifier_name_list), len(data_set_list), plot_location)
        for index, al_protocol in enumerate(al_protocol_list):
            if plot_type == 'tau':
                tau_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_tau_' + rankMetric + '.pickle'
                tau_list = pickle.load(open(tau_file_name, 'rb'))
                auc = trapz(tau_list, dx=10)
                plt.axvline(x=prevalence_ratio[datasource] * 100, linewidth=3, color='darkcyan')
                plt.plot(x_labels_set, tau_list, color_list[index], marker=marker_list[index], label= al_protocol+', AUC:' + str(auc)[:4],
                         linewidth=2.0)
            elif plot_type == 'f1':
                topic_summary_info_file_name = data_path + "per_topic_summary_" + seed_selection_type + "_" + classifier_name + "_" + al_protocol + '.pickle'
                topic_summary_info = pickle.load(open(topic_summary_info_file_name, 'rb'))
                f1_list = []

                # topic_summary_info[0] contains sum of f1 for all 50 topics
                for k in sorted(topic_summary_info[0].iterkeys()):
                    if k == 0: # skiping 0 value because it is only initial seed set of 10
                        continue
                    f1_list.append(topic_summary_info[0][k]/float(number_of_topic))
                auc = trapz(f1_list, dx=10)
                plt.plot(x_labels_set, f1_list, color_list[index], marker=marker_list[index],
                         label=al_protocol + ', AUC:' + str(auc)[:4],
                         linewidth=2.0)
            elif plot_type == 'drop' or plot_type == 'mismatch':
                system_drop_list = []
                original_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_original_' + rankMetric + '.pickle'
                original_system_metric_value = pickle.load(open(original_system_metric_value_file_name, 'rb'))
                for i in xrange(1, len(train_per_centage)):
                    predicted_system_metric_value_file_name = data_path + seed_selection_type + '_' + classifier_name + '_' + al_protocol + '_predicted_' + rankMetric + '_' + str(
                        i) + '.pickle'
                    predicted_system_metric_value = pickle.load(open(predicted_system_metric_value_file_name, 'rb'))
                    if plot_type == 'drop':
                        system_drop_list.append(drop_calculator(original_system_metric_value, predicted_system_metric_value)[0])
                    elif plot_type == 'mismatch':
                        system_drop_list.append(
                            drop_calculator(original_system_metric_value, predicted_system_metric_value)[1])

                auc = trapz(system_drop_list, dx=10)
                '''
                plt.plot(x_labels_set, system_drop_list, color_list[index], marker=marker_list[index],
                         label=al_protocol + ', AUC:' + str(auc)[:4],
                         linewidth=2.0)
                '''
                plt.plot(x_labels_set, system_drop_list, color_list[index], marker=marker_list[index],
                         label=al_protocol, linewidth=2.0)

        if plot_type == 'tau':
            if plot_location%2!=0:
                #plt.ylabel('Classifier = '+ classifier_name +'\ntau correlation\nusing '+rankMetric, size=16)
                plt.ylabel('tau correlation\nusing ' + rankMetric, size=16)

            plt.ylim([0.6, 1])
            plt.yticks(np.arange(0.7, 1.0, 0.1))
        elif plot_type == 'f1':
            if plot_location % 2 != 0:
                #plt.ylabel('Classifier = ' + classifier_name + "\nF1", size=16)
                plt.ylabel("F1", size=16)

            plt.ylim([0.4, 1])
            plt.yticks(np.arange(0.4, 1.0, 0.1))
            #plt.ylim([min(f1_list), 1])
            #plt.yticks(np.arange(min(f1_list),1.0,0.1))
        elif plot_type == 'drop':
            if plot_location % 2 != 0:
                plt.ylabel('Classifier = ' + classifier_name + "\nMax drop in a system's rank"+"\nusing "+rankMetric, size=9)
            plt.ylim([0, 50])
            plt.yticks(np.arange(0, 50, 5))
        elif plot_type == 'mismatch':
            if plot_location % 2 != 0:
                plt.ylabel('Classifier = ' + classifier_name + "\nNumber of mismatch in systems' rank"+"\nusing "+rankMetric, size=9)
            plt.ylim([0, 90])
            plt.yticks(np.arange(0, 90, 10))
        print prevalence_ratio[datasource]*100

        plt.xlabel('% of human judgments', size=16)
        plt.title(data_set_name_list[data_set_name_index], size=16)
        plt.grid(linestyle='dotted')
        plt.xticks(x_labels_set, x_labels_set)
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
        plot_location = plot_location + 1

plt.tight_layout()
if plot_type == 'tau':
    plt.savefig(plot_address + seed_selection_type + '_' + plot_type + '_' + rankMetric +'.pdf', format='pdf')
elif plot_type == 'f1' or plot_type == 'drop' or plot_type == 'mismatch':
    plt.savefig(plot_address + seed_selection_type + '_' + plot_type +'.pdf', format='pdf')

