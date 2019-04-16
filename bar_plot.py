from numpy import trapz
import os
import matplotlib
import operator
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys

import matplotlib.gridspec as gridspec
from numpy import trapz

gs = gridspec.GridSpec(5, 2)

import pickle
import numpy as np


budget_increment = 500
datasetsize = 14000

folder = 0
last_budget = int(datasetsize/1000)*1000

datalist = ['TREC8', 'WT2013','WT2014']
datasource = 'WT2014'
protocol = 'CAL'
topic_sampling_protocol = ['MAB', 'serial']

print "last budget", last_budget

budget_list = []

# budget list from CIKM 2018 paper for TREC 8
budget_list_TREC8 = [10134, 19231, 28125, 36947, 45584, 54071, 62476, 70654, 78805]

for budget in xrange(3000, 14000, budget_increment):
    budget_list.append(budget)


budget_list = sorted(budget_list)
found_per_budget = {}


for budget in budget_list:

    for datasource in datalist:
        topic_related_file = "/work/04549/mustaf/lonestar/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/MAB/"+str(folder)+"/" + datasource + "_related_documents_per_topic.pickle"
# is a dictionary topicString is the key, value is the number of related document
topic_related_documents = pickle.load(open(topic_related_file, "rb"))

sum_r = 0
for k, v in topic_related_documents.iteritems():
    sum_r += v
    #print k, v

print sum_r

#for budget_limit in budget_list:
#    print budget_limit



#budget_list = [2000, 2500]

'''
a = pickle.load(open("/work/04549/mustaf/lonestar/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/MAB/"+str(folder)+"/budget_protocol:"+str(protocol)+"_batch:1_seed:10.pickle", "rb"))
for key in sorted(a.iterkeys()):
    print key, a[key]

a = pickle.load(open("/work/04549/mustaf/lonestar/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/MAB/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_" + str(2500) + "_budget_prf.txt", "rb"))
for key in sorted(a.iterkeys()):
    for x in a[key]:
        if x[2] < 1.0:
            print key, x

'''

sum = 0


print len(budget_list)

budget_list = [10000]

def folder_no(datasource, topic_protocol):
    if datasource == 'TREC8' and topic_protocol == 'MAB':
        folder = '5'
    elif datasource == 'TREC8' and topic_protocol == 'serial':
        folder = '11'
    elif datasource == 'WT2013' and topic_protocol == 'MAB':
        folder = '10'
    elif datasource == 'WT2013' and topic_protocol == 'serial':
        folder = '5'
    elif datasource == 'WT2014' and topic_protocol == 'MAB':
        folder = '10'
    elif datasource == 'WT2014' and topic_protocol == 'serial':
        folder = '0'

    return  folder

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5.5))
var = 1
for budget in budget_list:

    for datasource in datalist:

        topic_all = {}

        topic_protocol = 'MAB'
        folder = folder_no(datasource,topic_protocol)

        topic_related_file = "/work/04549/mustaf/lonestar/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/"+str(topic_protocol)+"/" + str(
            folder) + "/" + datasource + "_related_documents_per_topic.pickle"

        topic_non_related_file = None
        if datasource == 'TREC8':
            topic_non_related_file = '/work/04549/mustaf/lonestar/data/TREC/deterministic1/TREC8/result/ranker/oversample/serial/0/TREC8_non_related_documents_per_topic.pickle'
        elif datasource == 'WT2013':
            topic_non_related_file = '/work/04549/mustaf/lonestar/data/TREC/deterministic1/WT2013/result/ranker/oversample/serial/0/WT2013_non_related_documents_per_topic.pickle'
        elif datasource == 'WT2014':
            topic_non_related_file = '/work/04549/mustaf/lonestar/data/TREC/deterministic1/WT2014/result/ranker/oversample/serial/0/WT2014_non_related_documents_per_topic.pickle'


        # is a dictionary topicString is the key, value is the number of related document
        topic_related_documents = pickle.load(open(topic_related_file, "rb"))
        topic_non_related_documents = pickle.load(open(topic_non_related_file,"rb"))

        topic_judged = {}
        for key in sorted(topic_related_documents.iterkeys()):
            topic_judged[key] = topic_related_documents[key] + topic_non_related_documents[key]

        import operator

        sorted_x = sorted(topic_judged.items(), key=operator.itemgetter(1))
        #print sorted_x

        mab = pickle.load(open("/work/04549/mustaf/lonestar/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/"+str(topic_protocol)+"/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))

        topic_protocol = 'serial'
        folder = folder_no(datasource,topic_protocol)

        serial =  pickle.load(open("/work/04549/mustaf/lonestar/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/"+str(topic_protocol)+"/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))


        # a list of tuples (topicId, related, non-related)
        b = mab[budget]
        c = serial[budget]

        bar_list = []
        topic_name = []
        mab_list = []
        serial_list = []

        for sx in sorted_x:
            bar_list.append(sx[1])
            topic_name.append(sx[0])
            for x in b:
                if x[0] == sx[0]:
                    mab_list.append(x[1]+x[2])
                    break

            for y in c:
                if y[0] == sx[0]:
                    serial_list.append(y[1]+y[2])
                    break

        plt.subplot(1, 3, var)
        var = var + 1
        plt.bar(range(len(bar_list)),bar_list, alpha=0.3, color='b', label = 'Ground Truth')
        plt.plot(range(len(bar_list)), mab_list, '-r', marker='o', label='MAB')
        plt.plot(range(len(bar_list)), serial_list, '-g', marker='D', label='RR')

        plt.title(datasource)
        plt.xticks(range(len(bar_list)),topic_name, rotation='vertical', fontsize = 5)
        plt.xlabel("Topic Id")
        plt.ylabel("Number of judged documents")
        plt.legend(loc='best', fontsize=10)
        plt.title(datasource)
        #plt.grid()

        print datasource, np.sum(bar_list), np.sum(mab_list), np.sum(serial_list)

plt.tight_layout()
print os.getcwd()
#plt.savefig(os.getcwd() + "/bar_plot_all.pdf", format='pdf')
plt.savefig(os.getcwd() + "/bar_plot_all.png", format='png')
