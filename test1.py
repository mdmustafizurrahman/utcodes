import pickle
import numpy as np
import math


budget_increment = 500
#datasetsize = 13420
#datasetsize = 14000
datasetsize = 86830

folder = 5
last_budget = int(datasetsize/1000)*1000

datasource = 'TREC8'
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

topic_related_file = "/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/MAB/"+str(folder)+"/" + datasource + "_related_documents_per_topic.pickle"
# is a dictionary topicString is the key, value is the number of related document
topic_related_documents = pickle.load(open(topic_related_file, "rb"))

sum_r = 0
for k, v in topic_related_documents.iteritems():
    sum_r += v
    #print k, v

print "N--->", sum_r

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


# RMSE calculation
#budget_list = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]
for budget in budget_list:
    for protocol in protocol_list:
        sum_m = 0
        a_m = pickle.load(open(
            "/work/04549/mustaf/maverick/data/TREC/deterministic1/" + datasource + "/result/ranker/oversample/MAB/" + str(
                folder) + "/prediction_protocol:" + str(protocol) + "_batch:1_seed:10_fold1_" + str(
                budget) + "_budget_detail.txt", "rb"))

        # x[0] topicId 401
        # x[1] denots # of relevant documents for topic mentioned in x[0] at budget
        for x in a_m[budget]:
            sum_m += (x[1] - topic_related_documents[x[0]])*(x[1] - topic_related_documents[x[0]])

        number_of_topics = len(topic_related_documents)
        sum_m = float(sum_m) / float(number_of_topics)

        print "Budget = ", str(budget) , ", RMSE = ", math.sqrt(sum_m)

#exit(0)



for budget in budget_list:
    #print str(budget),
    for protocol in protocol_list:

        sum = 0
        sum_d = 0
        sum_m = 0

        '''
        a = pickle.load(open("/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/oracle/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))
        

        oracle_list = []
        recall = 0
        for x in a[budget]:
            sum+= x[1]
            recall = float(x[1])/float(topic_related_documents[x[0]])
            oracle_list.append(recall)

        '''
        recall = 0


        mab_lsit = []
        a_m = pickle.load(open("/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/MAB/"+str(folder)+"/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_"+str(budget)+"_budget_detail.txt","rb"))
        sum_m = 0

        for x in a_m[budget]:
            sum_m+= x[1]
            recall = float(x[1]) / float(topic_related_documents[x[0]])

            mab_lsit.append(recall)

        recall = 0

        '''
        serial_lsit = []
        a_d = pickle.load(open(
            "/work/04549/mustaf/maverick/data/TREC/deterministic1/"+ datasource +"/result/ranker/oversample/serial/5/prediction_protocol:"+str(protocol)+"_batch:1_seed:10_fold1_" + str(
                budget) + "_budget_detail.txt", "rb"))


        for x in a_d[budget]:
            sum_d += x[1]
            recall = float(x[1]) / float(topic_related_documents[x[0]])

            serial_lsit.append(recall)
        '''
        print "protocol = "+str(protocol)+ ",budget = ", str(budget),"per topic avg budget =", str(budget/50.0),", MAB = ", str(sum_m), ", avg=", str(sum_m/50.0)[:4]
        #print "budget = ", str(budget), ", MAB = ", str(np.std(mab_lsit)), ", Oracle = ",str(np.std(oracle_list)) , ", serial = ", str(np.std(serial_lsit))
        #print "budget = ", str(budget), ", MAB = ", str(np.mean(mab_lsit)), ", Oracle = ", str(
        #    np.mean(oracle_list)), ", serial = ", str(np.mean(serial_lsit))
        '''
        if sum > sum_m and sum > sum_d:
            print "&\\textbf{", str(sum), "}&", str(sum_m), "&", str(sum_d),
        if sum_m > sum and sum_m > sum_d:
            print "&", str(sum), "&\\textbf{", str(sum_m), "}&", str(sum_d),
        if sum_d > sum  and sum_d > sum_m:
            print "&", str(sum), "&", str(sum_m), "&\\textbf{", str(sum_d),"}",
        
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

    #print "\\\\"
    #print "\\hline"



'''
folder = 0
protocol = 'CAL'
import matplotlib
import operator
matplotlib.use('Agg')

import matplotlib.pyplot as plt

budget_list = [8000]

marker = ['bo', 'p','o']

topic_list = []
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
'''