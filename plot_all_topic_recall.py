import matplotlib
matplotlib.use('Agg')
import seaborn
import pickle
import numpy as np

'''
import matplotlib
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import trapz
gs = gridspec.GridSpec(5, 2)
import seaborn

budget_increment = 2000
#datasetsize = 13420
datasetsize = 14000
#datasetsize = 86830


last_budget = int(datasetsize/1000)*1000

#datasource = 'WT2014'
dataset_list = ['WT2014','WT2013']
protocol_list = ['CAL','SAL','SPL']
run_list = [10,11,12]
print "last budget", last_budget

budget_list = []
budget_limit_to_train_percentage_mapper = {}

# budget list from CIKM 2018 paper for TREC 8
budget_list_TREC8 = [10134, 19231, 28125, 36947, 45584, 54071, 62476, 70654, 78805]

for budget in xrange(3000, last_budget, budget_increment):
    budget_list.append(budget)

'''
if datas == 'TREC8':
    for budget in budget_list_TREC8:
        budget_list.append(budget)
        budget_limit_to_train_percentage_mapper[budget] = budget
'''
budget_list = sorted(budget_list)

found_per_budget = {}

conds = ['RR', 'Oracle', 'MAB']

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

colors = seaborn.color_palette('colorblind')
markers = ['s', 'o', '^', 'v', '<']

print len(budget_list)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 5.5))
#fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(5, 5.5))

var = 1
for datasource in dataset_list:
    topic_related_file = "/work/04549/mustaf/lonestar/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/MAB/10/" + datasource + "_related_documents_per_topic.pickle"
    # is a dictionary topicString is the key, value is the number of related document


    topic_related_documents = pickle.load(open(topic_related_file, "rb"))

    total_related = 0
    for key, val in topic_related_documents.iteritems():
        total_related = total_related + val

    #plt.close()
    for protocol in protocol_list:
        # all the following list are a list of list
        # index is 0 -> first budget in budget list
        # value -> list of recall value per topic in that budget
        serial_data = []
        oracle_data = []
        mab_data    = []

        for folder in run_list:
            oracle_list = []
            serial_lsit = []
            mab_lsit = []
            for budget in budget_list:
                a = pickle.load(open("/work/04549/mustaf/lonestar/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/smartoracle/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))

                sum = 0
                sum_d = 0
                sum_m = 0

                for x in a[budget]:
                    sum+= x[1]

                recall = float(sum)/float(total_related)
                oracle_list.append(recall)



                a_m = pickle.load(open("/work/04549/mustaf/lonestar/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/MAB/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))
                sum_m = 0

                for x in a_m[budget]:
                    sum_m+= x[1]

                recall = float(sum_m) / float(total_related)
                mab_lsit.append(recall)

                recall = 0


                # for serial folder is always 0
                a_d = None
                if datasource == "WT2014":
                    a_d = pickle.load(open("/work/04549/mustaf/lonestar/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/serial/0/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_" + str(budget) + "_budget_detail.txt", "rb"))
                elif datasource == "WT2013":
                    a_d = pickle.load(open("/work/04549/mustaf/lonestar/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/serial/5/prediction_protocol:" + str(protocol) + "_batch:1_seed:10_fold1_" + str(budget) + "_budget_detail.txt", "rb"))

                for x in a_d[budget]:
                    sum_d += x[1]

                recall = float(sum_d) / float(total_related)
                serial_lsit.append(recall)

            serial_data.append(serial_lsit)
            oracle_data.append(oracle_list)
            mab_data.append(mab_lsit)

        serial_auc = trapz(serial_lsit, dx=20)
        oracle_auc = trapz(oracle_list, dx=20)
        mab_auc = trapz(mab_lsit, dx=20)

        plt.subplot(2, 3, var)
        seaborn.axes_style("darkgrid")
        seaborn.tsplot(data=oracle_data, time=range(len(budget_list)), condition=conds[1] + ", AUC:"+str(oracle_auc)[:4],
                       color=colors[1], marker=markers[1], markersize=5)
        seaborn.tsplot(data=mab_data, time=range(len(budget_list)), condition=conds[2] + ", AUC:"+str(mab_auc)[:4],color = colors[2], marker = markers[2], markersize = 5)
        seaborn.tsplot(data=serial_data, time=range(len(budget_list)), condition=conds[0] + ", AUC:"+str(serial_auc)[:4], color=colors[0],
                       marker=markers[0], markersize=5)

        if var == 1:
            plt.title("CAL")
        elif var == 2:
            plt.title("SAL")
        elif var == 3:
            plt.title("SPL")

        if var == 1:
            plt.ylabel("WT2014"+"\n"+'Recall')

        if var == 4:
            plt.ylabel("WT2013"+"\n"+'Recall')
        plt.grid(b=True,linestyle='-')

        plt.yticks(size=7)
        if var >=4:
            plt.xlabel("Total Budget")
            plt.xticks(range(len(budget_list)), budget_list,size=7)
        else:
            plt.xticks(range(len(budget_list)), [], rotation='vertical')

        plt.legend(loc='best', fontsize=10)
        plt.ylim([0.2, 1])

        var = var + 1

#plt.xlabel("Allocated budget")
#plt.ylabel("Recall")
plt.tight_layout()
plt.savefig("all_topic_recall_v1.png", bbox_inches = 'tight', pad_inches = 0.1, type='png')
