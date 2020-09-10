import matplotlib
matplotlib.use('Agg')
import seaborn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn

budget_increment = 1000
#datasetsize = 13420
datasetsize = 14000
#datasetsize = 86830


last_budget = int(datasetsize/1000)*1000

datasource = 'WT2014'
dataset_list = ['WT2014']
protocol_list = ['CAL']
run_list = [0]
print "last budget", last_budget

budget_list = []
budget_limit_to_train_percentage_mapper = {}

# budget list from CIKM 2018 paper for TREC 8
budget_list_TREC8 = [10134, 19231, 28125, 36947, 45584, 54071, 62476, 70654, 78805]

for budget in xrange(2000, last_budget, budget_increment):
    budget_list.append(budget)

if datasource == 'TREC8':
    for budget in budget_list_TREC8:
        budget_list.append(budget)
        budget_limit_to_train_percentage_mapper[budget] = budget

budget_list = sorted(budget_list)

found_per_budget = {}

conds = ['Round Robin', 'Oracle', 'MAB']

'''
a = pickle.load(open("/work/04549/mustaf/maverick/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/MAB/"+str(folder)+"/budget_protocol:"+str(protocol)+"_batch:1_seed:10.pickle", "rb"))
for key in sorted(a.iterkeys()):
    print key, a[key]

a = pickle.load(open("/work/04549/mustaf/maverick/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/MAB/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_" + str(2500) + "_budget_prf.txt", "rb"))
for key in sorted(a.iterkeys()):
    for x in a[key]:
        if x[2] < 1.0:
            print key, x

'''

sum = 0

colors = seaborn.color_palette('colorblind')
markers = ['s', 'o', '^', 'v', '<']

print len(budget_list)
folder = 0
for datasource in dataset_list:
    topic_related_file = "/work/04549/mustaf/maverick/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/serial/" + str(
        folder) + "/" + datasource + "_related_documents_per_topic.pickle"
    # is a dictionary topicString is the key, value is the number of related document
    topic_related_documents = pickle.load(open(topic_related_file, "rb"))
    plt.close()
    for protocol in protocol_list:
        # all the following list are a list of list
        # index is 0 -> first budget in budget list
        # value -> list of recall value per topic in that budget
        serial_data = []
        oracle_data = []
        mab_data    = []

        for budget in budget_list:
            a = pickle.load(open("/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/oracle/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))
            sum = 0
            sum_d = 0
            sum_m = 0

            oracle_list = []
            recall = 0
            for x in a[budget]:
                sum+= x[1]
                recall = float(x[1])/float(topic_related_documents[x[0]])
                oracle_list.append(recall)

            recall = 0


            mab_lsit = []
            a_m = pickle.load(open("/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/MAB/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))
            sum_m = 0

            for x in a_m[budget]:
                sum_m+= x[1]
                recall = float(x[1]) / float(topic_related_documents[x[0]])
                mab_lsit.append(recall)

            recall = 0



            serial_lsit = []
            a_d = pickle.load(open(
                "/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/serial/0/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_" + str(
                    budget) + "_budget_detail.txt", "rb"))


            for x in a_d[budget]:
                sum_d += x[1]
                recall = float(x[1]) / float(topic_related_documents[x[0]])
                serial_lsit.append(recall)

            serial_data.append(serial_lsit)
            oracle_data.append(oracle_list)
            mab_data.append(mab_lsit)

        seaborn.tsplot(data=np.array(serial_data).T.tolist(), time=range(len(budget_list)), condition=conds[0], color = colors[0], marker = markers[0], markersize = 5)
        seaborn.tsplot(data=np.array(mab_data).T.tolist(), time=range(len(budget_list)), condition=conds[2],color = colors[2], marker = markers[2], markersize = 5)
        seaborn.tsplot(data=np.array(oracle_data).T.tolist(), time=range(len(budget_list)), condition=conds[1],
                       color=colors[1], marker=markers[1], markersize=5)
plt.grid()
plt.xticks(range(len(budget_list)), budget_list, rotation='vertical')
plt.xlabel("Allocated budget")
plt.ylabel("Recall")
plt.legend(loc ='best', fontsize=5)
plt.savefig("fig.pdf", bbox_inches = 'tight', pad_inches = 0.1, type='pdf')
