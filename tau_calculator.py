from scipy.stats.stats import kendalltau
from numpy import trapz
import os
import matplotlib
import operator
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
os.chdir('/work/04549/mustaf/lonestar/data/TREC/trec_eval.9.0')

base_address1 = "/work/04549/mustaf/lonestar/data/TREC/"
plotAddress =  "/work/04549/mustaf/lonestar/data/TREC/plot/"

dataset_list = [sys.argv[1]]
protocol_list = ['CAL']
topic_sampling_protocol = sys.argv[2] #serial, smartoracle, MAB
#map_list = ['P.10'] # map, P.10, infAP
#map_list = ['map'] # map, P.10, infAP
#map_list = ['infAP'] # map, P.10, infAP
map_list = ['gm_map']

budget_increment = 500
useBudget = True
useBLAparam = True
BLAparams = 1


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
rnus_not_in_WT_2014_adhoc = ['UDInfoWebRiskTR','UDInfoWebRiskRM','UDInfoWebRiskAX','ICTNET14RSR1','ICTNET14RSR2','uogTrq1','uogTrBwf','ICTNET14RSR3','udelCombCAT2','uogTrDwsts','wistud.runD','wistud.runE']

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
budget_list_TREC8 = range(2000,14000,budget_increment)


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
stringUse = ''

for stringUse in map_list:
    for use_ranker in ranker_list:
        for iter_sampling in sampling_list:
            s = ""
            s1 = ""
            originalqrelMap = []
            predictedqrelMap = []
            for datasource in dataset_list:  # 1
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
                        predictionAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample/' + topic_sampling_protocol + '/20/BLAparams_' + str(
                            BLAparams) + '/'
                        predictionModifiedAddress = base_address1 + 'deterministic1/' + datasource + '/result/ranker/oversample' + topic_sampling_protocol + '/20/BLAparams_' + str(
                            BLAparams) + '/'

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

                print "Original Part"


                ######################

                fileList = sorted(os.listdir(originAdress))
                for fileName in fileList:
                    if datasource == 'TREC8' and fileName not in run_name:
                        continue
                    system = originAdress + fileName
                    #shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
                    shellCommand = './trec_eval -m '+stringUse+' ' + qrelAdress + ' ' + system

                    print shellCommand
                    import subprocess
                    p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in p.stdout.readlines():
                        print line
                        values = line.split()
                        map = float(values[2])
                        originalqrelMap.append(map)

                    retval = p.wait()


                originalMapResult = originalMapResultBase + stringUse +'_of_all_runs_using_original_qrels.txt'

                import pickle
                pickle.dump(originalqrelMap, open(originalMapResult, "wb"))

                originalMapResult = originalMapResultBase + stringUse + '_of_all_runs_using_original_qrels.txt'
                originalqrelMap = pickle.load(open(originalMapResult, "rb"))

                print len(originalqrelMap)

                #exit(0)

                tau_list = {} # budget is the key, value is the tau
                ##############################

                if useBudget == True:
                    for budget in budget_list:
                        predictionQrel = ""

                        predictionQrel = predictionAddress + "prediction_protocol:CAL_batch:1_seed:10_fold1_"+str(budget)+"_human_qrels.txt"
                        print predictionQrel

                        predictionModifiedQrel = predictionAddress + "prediction_protocol:CAL_batch:1_seed:10_fold1_"+str(budget)+"_human_modified.txt"

                        # reading the prediction file and modifyinh it insert one extra column at position 2
                        f = open(predictionQrel)
                        tmpstring = ""
                        for lines in f:
                            values = lines.split()
                            tmpstring = tmpstring + values[0] + " 0 "+values[1]+" "+values[2]+"\n"

                        text_file = open(predictionModifiedQrel, "w")
                        text_file.write(tmpstring)
                        text_file.close()


                        fileList = sorted(os.listdir(originAdress))
                        for fileName in fileList:
                            if datasource == 'TREC8' and fileName not in run_name:
                                continue
                            system = originAdress + fileName
                            # shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
                            shellCommand = './trec_eval -m '+stringUse +' '+ predictionModifiedQrel + ' ' + system

                            print shellCommand
                            import subprocess
                            p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT)
                            for line in p.stdout.readlines():
                                print line
                                values = line.split()
                                map = float(values[2])
                                predictedqrelMap.append(map)

                            retval = p.wait()

                        predictionMapResult = predictionAddress + "prediction_protocol:CAL_batch:1_seed:10_fold1_"+str(budget)+"_human_"+stringUse+".pickle"
                        pickle.dump(predictedqrelMap, open(predictionMapResult, "wb"))

                        predictedqrelMap = pickle.load(open(predictionMapResult, "rb"))
                        #print len(predictedqrelMap)
                        print len(originalqrelMap), len(predictedqrelMap)
                        tau, p_value = kendalltau(originalqrelMap, predictedqrelMap)
                        tau_list[budget] = tau

                        predictedqrelMap = []  # cleaning it for next trains_percenatge

                else:
                    predictionAddress = "/work/04549/mustaf/lonestar/data/TREC/TREC8/topKPoolQrels/"
                    for pool in range(10,100,10):
                        predictionModifiedQrel = ""
                        predictionModifiedQrel = predictionAddress + "pool_"+str(pool)+".txt"
                        print predictionModifiedQrel

                        fileList = sorted(os.listdir(originAdress))
                        for fileName in fileList:
                            if fileName not in run_name:
                                continue
                            system = originAdress + fileName
                            # shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
                            shellCommand = './trec_eval -m ' + stringUse + ' ' + predictionModifiedQrel + ' ' + system

                            print shellCommand
                            import subprocess

                            p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT)
                            for line in p.stdout.readlines():
                                print line
                                values = line.split()
                                map = float(values[2])
                                predictedqrelMap.append(map)

                            retval = p.wait()

                        predictionMapResult = predictionAddress + "pool_"+str(pool)+"_"+stringUse + ".pickle"
                        pickle.dump(predictedqrelMap, open(predictionMapResult, "wb"))

                        predictedqrelMap = pickle.load(open(predictionMapResult, "rb"))
                        # print len(predictedqrelMap)
                        print len(originalqrelMap), len(predictedqrelMap)
                        tau, p_value = kendalltau(originalqrelMap, predictedqrelMap)
                        tau_list[pool] = tau

                        predictedqrelMap = []  # cleaning it for next trains_percenatge

                for key in sorted(tau_list.iterkeys()):
                    print key, tau_list[key]

                predictiontauResult = predictionAddress + "all_pool_tau_on_"+stringUse+".pickle"

                pickle.dump(tau_list, open(predictiontauResult, "wb"))

                exit(0)

                print len(training_variation)

                auc_SAL = trapz(protocol_result['SAL'], dx=10)
                auc_CAL = trapz(protocol_result['CAL'], dx=10)
                auc_SPL = trapz(protocol_result['SPL'], dx=10)

                print auc_SAL, auc_CAL, auc_SPL
                print var
                plt.subplot(2,4,var)

                plt.plot(x_labels_set, protocol_result['CAL'], '-b', marker='^', label='CAL, AUC:' + str(auc_CAL)[:4],
                         linewidth=2.0)

                plt.plot(x_labels_set, protocol_result['SAL'], '-r', marker='o', label='SAL, AUC:'+str(auc_SAL)[:4], linewidth=2.0)
                plt.plot(x_labels_set, protocol_result['SPL'], '-g', marker='D', label='SPL, AUC:'+str(auc_SPL)[:4], linewidth=2.0)



                if var == 1:
                    plt.ylabel('tau correlation \n using MAP',size = 16)
                if var == 5:
                    plt.ylabel('tau correlation \n using bpref', size=16)

                if var >=5:
                    plt.xlabel('% of human judgments', size=16)

                plt.ylim([0.7, 1])
                plt.yticks([0.7, 0.8, .9, 1.0])
                plt.legend(loc=4)
                param = "($\\alpha$ = " + str(alpha_param) + ", $\lambda$ = " + str(lambda_param) + ")"
                if datasource == 'gov2':
                    plt.title('TB\'06 ' + param, size=16)
                elif datasource == 'WT2013':
                    plt.title('WT\'13 ' + param, size=16)
                elif datasource == 'WT2014':
                    plt.title('WT\'14 ' + param, size=16)
                else:
                    plt.title('Adhoc\'99 ' + param, size=16)
                plt.grid()
                var = var + 1

#plt.suptitle(s1, size=16)
plt.tight_layout()
plt.savefig(plotAddress + s1 + 'map_bpref_crowd.pdf', format='pdf')





