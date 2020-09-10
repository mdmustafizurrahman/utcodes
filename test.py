import pickle
import numpy as np


budget_increment = 500
#datasetsize = 13420
datasetsize = 14000
#datasetsize = 86830

folder = 0
last_budget = int(datasetsize/1000)*1000

datasource = 'WT2014'
#protocol_list = ['CAL', 'SAL', 'SPL']
protocol_list = ['CAL']

print "last budget", last_budget



budget_list = []
budget_limit_to_train_percentage_mapper = {}

# budget list from CIKM 2018 paper for TREC 8
budget_list_TREC8 = [10134, 19231, 28125, 36947, 45584, 54071, 62476, 70654, 78805]


for budget in xrange(2000, last_budget, budget_increment):
    budget_list.append(budget)
    budget_limit_to_train_percentage_mapper[budget] = budget



if datasource == 'TREC8':
    for budget in budget_list_TREC8:
        budget_list.append(budget)
        budget_limit_to_train_percentage_mapper[budget] = budget

budget_list = sorted(budget_list)
budget_limit_to_train_percentage_mapper = sorted(budget_limit_to_train_percentage_mapper)

print len(budget_list)

# budget_list.append(last_budget+budget_increment)
# budget_limit_to_train_percentage_mapper[last_budget+budget_increment] = 1.01
found_per_budget = {}

topic_related_file = "/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/serial/"+str(folder)+"/" + datasource + "_related_documents_per_topic.pickle"
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


print len(budget_list)

for budget in budget_list:
    print str(budget),
    for protocol in protocol_list:
        a = pickle.load(open("/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/smartoracle/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))
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

        #print "protocol = "+str(protocol)+ ",budget = ", str(budget),", MAB = ", str(sum_m), ", Oracle = ", str(sum), ", serial = ", str(sum_d)
        #print "budget = ", str(budget), ", MAB = ", str(np.std(mab_lsit)), ", Oracle = ",str(np.std(oracle_list)) , ", serial = ", str(np.std(serial_lsit))
        #print "budget = ", str(budget), ", MAB = ", str(np.mean(mab_lsit)), ", Oracle = ", str(
        #    np.mean(oracle_list)), ", serial = ", str(np.mean(serial_lsit))

        if sum > sum_m and sum > sum_d:
            print "&\\textbf{", str(sum), "}&", str(sum_m), "&", str(sum_d),
        if sum_m > sum and sum_m > sum_d:
            print "&", str(sum), "&\\textbf{", str(sum_m), "}&", str(sum_d),
        if sum_d > sum  and sum_d > sum_m:
            print "&", str(sum), "&", str(sum_m), "&\\textbf{", str(sum_d),"}",
        '''
        std_o = str(np.std(oracle_list))[:5]
        std_m = str(np.std(mab_lsit))[:5]
        std_d = str(np.std(serial_lsit))[:5]

        if std_o < std_m and std_o < std_d:
            print "&\\textbf{", str(std_o), "}&", str(std_m), "&", str(std_d),

        if std_m < std_o and std_m < std_d:
            print "&", str(std_o), "&\\textbf{", str(std_m), "}&", str(std_d),
        if std_d < std_o  and std_d < std_m:
            print "&", str(std_o), "&", str(std_m), "&\\textbf{", str(std_d),"}",
        '''


    print "\\\\"
    print "\\hline"


folder = 0
protocol = 'CAL'
import matplotlib
import operator
matplotlib.use('Agg')

import matplotlib.pyplot as plt

budget_list = [8000]

marker = ['bo', 'p','o']

topic_list = []
t1 = []
topic_allocation = []
for i, budget in enumerate(budget_list):

    a_m = pickle.load(open(
        "/work/04549/mustaf/maverick/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/MAB/" + str(
            folder) + "/prediction_protocol:" + str(protocol) + "_batch:1_seed:10_fold1_" + str(
            budget) + "_budget_detail.txt", "rb"))

    topic_dic = {}
    for x in a_m[budget]:
        topic_dic[int(x[0])] = int(x[1]) + int(x[2])

    topic_sorted_list = sorted(topic_dic.items(), key=operator.itemgetter(1), reverse=True)

    topic_list = []
    topic_allocation = []

    for x in topic_sorted_list:
        topic_list.append(x[0])
        topic_allocation.append(x[1])

    t1 = np.arange(0, len(topic_list), 1)

    plt.plot(t1, topic_allocation, marker[i], linestyle='-', linewidth=2.0, label = "Total Budget = " + str(budget))
    #plt.plot(t1, topic_allocation, 'k')
    plt.grid(True)

t1 = np.arange(0, len(topic_list), 1)
plt.xticks(t1, topic_list, rotation='vertical')
plt.xlabel("Topic Id")
plt.ylabel("Budget")
plt.legend()
plt.savefig("t.pdf", type='pdf')


plt.close()

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
fmri = sns.load_dataset("fmri")

print fmri['signal']
ax = sns.lineplot(x=t1, y=topic_allocation, data=fmri)
plt.savefig("t1.pdf", type='pdf')
