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


#os.chdir('/work/04549/mustaf/lonestar/data/TREC/trec_eval.9.0')
base_address1 = "/work/04549/mustaf/lonestar/data/TREC/"
plotAddress =  "/work/04549/mustaf/lonestar/data/TREC/plot/"

dataset_list = ['TREC8']
protocol_list = ['CAL']
topic_sampling_protocol_list = ['smartoracle', 'MAB','serial']
#topic_sampling_protocol_list = ['MAB','serial']

#map_list = ['map','P.10'] # map, P.10, infAP

map_list = ['map'] # map, P.10, infAP

budget_increment = 500
useBudget = True
useBLAparam = False
BLAparams = 100

ht_estimation = False
crowd = False
ranker_list = ['True']
sampling_list = ['True']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # skiping seed part which is named as 0.1
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# TREC-8 Adhoc Run List
runs = "input.1 input.8manexT3D1N0 input.acsys8alo input.acsys8amn input.AntHoc1 input.apl8c221 input.apl8n input.att99atdc input.att99atde input.cirtrc82 input.CL99SD input.CL99XT input.disco1 input.Dm8Nbn input.Dm8TFbn input.Flab8as input.Flab8atdn input.fub99a input.fub99tf input.GE8ATDN1 input.ibmg99a input.ibmg99b input.ibms99a input.ibms99b input.ic99dafb input.iit99au1 input.iit99ma1 input.INQ603 input.INQ604 input.isa25 input.isa50 input.kdd8ps16 input.kdd8qe01 input.kuadhoc input.mds08a3 input.mds08a4 input.Mer8Adtd1 input.Mer8Adtd2 input.MITSLStd input.MITSLStdn input.nttd8ale input.nttd8alx input.ok8alx input.ok8amxc input.orcl99man input.pir9Aatd input.pir9Attd input.plt8ah1 input.plt8ah2 input.READWARE input.READWARE2 input.ric8dpx input.ric8tpx input.Sab8A1 input.Sab8A2 input.Scai8Adhoc input.surfahi1 input.surfahi2 input.tno8d3 input.tno8d4 input.umd99a1 input.unc8al32 input.unc8al42 input.UniNET8Lg input.UniNET8St input.UT810 input.UT813 input.uwmt8a1 input.uwmt8a2 input.weaver1 input.weaver2"
run_name = []
for run in runs.split(" "):
    run_name.append(run)


result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}

budget_list = []
#budget_list_TREC8 = [10134, 19231, 28125, 36947, 45584, 54071, 62476, 70654, 78805]
budget_list_TREC8 = range(2000,11500,budget_increment)


datasource = dataset_list[0]
datasetsize = 0
if datasource == 'TREC8':
    for budget in budget_list_TREC8:
        budget_list.append(budget)

else:

    if datasource == 'WT2013':
        datasetsize = 14474
    elif datasource == 'WT2014':
        datasetsize = 14432

    last_budget = int(datasetsize / 1000) * 1000
    print "last budget", last_budget

    for budget in xrange(2000, last_budget, budget_increment):
        budget_list.append(budget)

if datasource == 'WT2013':
    if 13500 in budget_list:
        budget_list.remove(13500)

budget_list = sorted(budget_list)
print budget_list

var = 1
stringUse = map_list[0]
markers = ['o', '^', 'D', '<']
colors = ['-b','-r', '-g']

label_list = ['Oracle', 'MAB', 'RR']
fig, ax = plt.subplots(nrows=len(map_list), ncols=len(dataset_list), figsize=(15, 5.5))

for stringUse in map_list:
    for datasource in dataset_list:  # 1
        protocol_result = {}
        print var


        for index, topic_sampling_protocol in enumerate(topic_sampling_protocol_list):

            #if datasource == 'TREC8' and topic_sampling_protocol == 'smartoracle':
            #    continue

            originalqrelMap = []
            predictedqrelMap = []
            if datasource == 'gov2':
                originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                #qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/qrels.tb06.top50.txt'
                qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modified_qreldocsgov2.txt'
                originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/'
                destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/prediction/"
                predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modifiedprediction/"
                alpha_param = 1
                train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                     1.0,]  # skiping seed part which is named as 0.1
                lambda_param = 0.75
                alpha_param = 2

            elif datasource == 'TREC8':
                originAdress = base_address1 + datasource + "/systemRankings/"

                qrelAdress = base_address1 + datasource + '/qrels.trec8.adhoc'
                originalMapResultBase = base_address1 + datasource + "/"

                predictionAddress = base_address1 +'deterministic1/TREC8/result/ranker/oversample/'+ topic_sampling_protocol +'/5/'
                predictionModifiedAddress = base_address1 +'deterministic1/TREC8/result/ranker/oversample'+ topic_sampling_protocol +'/5'

                if topic_sampling_protocol == 'serial':
                    predictionAddress = base_address1 + 'deterministic1/TREC8/result/ranker/oversample/' + topic_sampling_protocol + '/11/'
                    predictionModifiedAddress = base_address1 + 'deterministic1/TREC8/result/ranker/oversample' + topic_sampling_protocol + '/11'
                elif topic_sampling_protocol == 'smartoracle':
                    predictionAddress = base_address1 + 'deterministic1/TREC8/result/ranker/oversample/' + topic_sampling_protocol + '/5/BLAparams_1/'
                    predictionModifiedAddress = base_address1 + 'deterministic1/TREC8/result/ranker/oversample' + topic_sampling_protocol + '/5/BLAparams_1'

                lambda_param = 0.75
                alpha_param = 1
                train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                     1.0, ]  # skiping seed part which is named as 0.1

            elif datasource == 'WT2013':
                originAdress = base_address1 + datasource + "/systemRankings/"

                # since we are skipping topics
                qrelAdress = base_address1 + datasource + '/modified_qreldocs2013.txt'
                originalMapResultBase = base_address1 + datasource + "/"

                predictionAddress = base_address1 + 'deterministic1/'+ datasource +'/result/ranker/oversample/'+ topic_sampling_protocol +'/10/'
                predictionModifiedAddress = base_address1 + 'deterministic1/'+ datasource +'/result/ranker/oversample'+ topic_sampling_protocol +'/10'

                if topic_sampling_protocol == 'serial':
                    predictionAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample/' + topic_sampling_protocol + '/5/'
                    predictionModifiedAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample' + topic_sampling_protocol + '/5'

                lambda_param = 0.75
                alpha_param = 2
                train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                     1.0, ]  # skiping seed part which is named as 0.1


            else:

                originAdress = base_address1 + datasource + "/systemRankings/"

                # since we are skipping topics
                qrelAdress = base_address1 + datasource + '/modified_qreldocs2014.txt'

                originalMapResultBase = base_address1 + datasource + "/"

                predictionAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample/' + topic_sampling_protocol + '/10/'
                predictionModifiedAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample' + topic_sampling_protocol + '/10'

                if useBLAparam == True:
                    predictionAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample/' + topic_sampling_protocol + '/21/BLAparams_'+ str(BLAparams) +'/'
                    predictionModifiedAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample' + topic_sampling_protocol + '/21/BLAparams_'+ str(BLAparams) +'/'

                if topic_sampling_protocol == 'serial':
                    predictionAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample/' + topic_sampling_protocol + '/0/'
                    predictionModifiedAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample' + topic_sampling_protocol + '/0'

                train_per_centage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                     1.0, 1.1]  # skiping seed part which is named as 0.1

                if crowd == True:
                    alpha_param = 2
                    lambda_param = 1.0
                    qrelAdress = '/media/nahid/Windows8_OS/clueweb12/qrels/qrelsadhoc2014crowd.txt'

                else:
                    alpha_param = 2
                    lambda_param = 1.0

            originalMapResult = originalMapResultBase + stringUse +'_of_all_runs_using_original_qrels.txt'

            originalqrelMap = pickle.load(open(originalMapResult, "rb"))


            #exit(0)

            tau_list = {} # budget is the key, value is the tau

            predictiontauResult = predictionAddress + "all_pool_tau_on_"+stringUse+".pickle"
            print predictiontauResult
            tau_list = pickle.load(open(predictiontauResult,"rb"))

            tmp_list = []

            for key in range(3000, 41000, 1000):
                tmp_list.append(tau_list[key])

            protocol_result[topic_sampling_protocol] = tmp_list

            #print len(training_variation)



        plt.subplot(len(map_list), len(dataset_list), var)
        x_labels_set = range(3000, 41000, 1000)

        auc_mab = trapz(protocol_result['MAB'], dx=10)
        auc_rr = trapz(protocol_result['serial'], dx=10)

        #if datasource != 'TREC8':
        auc_oracle = trapz(protocol_result['smartoracle'], dx=10)
        plt.plot(x_labels_set, protocol_result['smartoracle'], '-b', marker='^', label= 'Oracle, AUC:' + str(auc_oracle)[:4], linewidth=2.0)

        plt.plot(x_labels_set, protocol_result['MAB'], '-r', marker='o', label='MAB, AUC:'+str(auc_mab)[:4], linewidth=2.0)
        plt.plot(x_labels_set, protocol_result['serial'], '-g', marker='D', label='RR, AUC:'+str(auc_rr)[:4], linewidth=2.0)
        plt.ylim([0.5, 1])
        plt.legend(loc=4, fontsize=10)
        if var <= 3:
            plt.title(datasource)
        plt.grid()
        plt.show()
        x_labels_set_ticks = range(3000, 41000, 2000)
        if var >= 4:
            plt.xlabel("Total Budget")
            plt.xticks(x_labels_set_ticks, x_labels_set_ticks)

        if var == 1 or var == 4:
            if stringUse == 'P.10':
                string = 'P@10'
            else:
                string = 'MAP'
            plt.ylabel("Kendall's tau\nusing " + string)
        var = var + 1
        print "var", var

#plt.suptitle(s1, size=16)
plt.tight_layout()
print os.getcwd()
plt.savefig(os.getcwd() + "/tau_plot_TREC8.pdf", format='pdf')





