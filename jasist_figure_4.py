from scipy.stats.stats import kendalltau
from numpy import trapz
import os
import pickle

import operator
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

os.chdir('/work/04549/mustaf/maverick/data/TREC/trec_eval.9.0')

base_address1 = "/work/04549/mustaf/maverick/data/TREC/deterministic1/"
plotAddress = base_address1
baseAddress = base_address1

qrelAddress = {}
qrelAddress['TREC8'] = '/work/04549/mustaf/maverick/data/TREC/TREC8/relevance.txt'
qrelAddress['gov2'] = '/work/04549/mustaf/maverick/data/TREC/gov2/modified_qreldocsgov2_jasist.txt'
qrelAddress['WT2013'] = '/work/04549/mustaf/maverick/data/TREC/WT2013/modified_qreldocs2013_jasist_new.txt'
qrelAddress['WT2014'] = '/work/04549/mustaf/maverick/data/TREC/WT2014/modified_qreldocs2014_jasist_new.txt'

systemAddress = {}
systemAddress['TREC8'] = '/work/04549/mustaf/maverick/data/TREC/TREC8/systemRankings/'
systemAddress['gov2'] = '/work/04549/mustaf/maverick/data/TREC/gov2/systemRankings/'
systemAddress['WT2013'] = '/work/04549/mustaf/maverick/data/TREC/WT2013/systemRankings/'
systemAddress['WT2014'] = '/work/04549/mustaf/maverick/data/TREC/WT2014/systemRankings/'

systemName = {}
systemName['TREC8'] = 'input.ibmg99b'
systemName['gov2'] = 'input.indri06AdmD'
systemName['WT2013'] = 'input.ICTNET13RSR2'
systemName['WT2014'] = 'input.Protoss'

originalMapLocation = {}
originalMapLocation['TREC8'] = '/work/04549/mustaf/maverick/data/TREC/TREC8/'
originalMapLocation['gov2'] = '/work/04549/mustaf/maverick/data/TREC/gov2/'
originalMapLocation['WT2013'] = '/work/04549/mustaf/maverick/data/TREC/WT2013/'
originalMapLocation['WT2014'] = '/work/04549/mustaf/maverick/data/TREC/WT2014/'


# TREC-8 Adhoc Run List
runs = "input.1 input.8manexT3D1N0 input.acsys8alo input.acsys8amn input.AntHoc1 input.apl8c221 input.apl8n input.att99atdc input.att99atde input.cirtrc82 input.CL99SD input.CL99XT input.disco1 input.Dm8Nbn input.Dm8TFbn input.Flab8as input.Flab8atdn input.fub99a input.fub99tf input.GE8ATDN1 input.ibmg99a input.ibmg99b input.ibms99a input.ibms99b input.ic99dafb input.iit99au1 input.iit99ma1 input.INQ603 input.INQ604 input.isa25 input.isa50 input.kdd8ps16 input.kdd8qe01 input.kuadhoc input.mds08a3 input.mds08a4 input.Mer8Adtd1 input.Mer8Adtd2 input.MITSLStd input.MITSLStdn input.nttd8ale input.nttd8alx input.ok8alx input.ok8amxc input.orcl99man input.pir9Aatd input.pir9Attd input.plt8ah1 input.plt8ah2 input.READWARE input.READWARE2 input.ric8dpx input.ric8tpx input.Sab8A1 input.Sab8A2 input.Scai8Adhoc input.surfahi1 input.surfahi2 input.tno8d3 input.tno8d4 input.umd99a1 input.unc8al32 input.unc8al42 input.UniNET8Lg input.UniNET8St input.UT810 input.UT813 input.uwmt8a1 input.uwmt8a2 input.weaver1 input.weaver2"
rnus_not_in_WT_2014_adhoc = ['UDInfoWebRiskTR','UDInfoWebRiskRM','UDInfoWebRiskAX','ICTNET14RSR1','ICTNET14RSR2','uogTrq1','uogTrBwf','ICTNET14RSR3','udelCombCAT2','uogTrDwsts','wistud.runD','wistud.runE']

run_name = []
for run in runs.split(" "):
    run_name.append(run)

preLoaded = True # False means we have to calcuate all rank metric from scratch
protocol_list = ['SAL','CAL', 'SPL']
#dataset_list = ['gov2']
dataset_list = ['WT2014','WT2013', 'gov2', 'TREC8']
ranker_list = ['False']
sampling_list = ['True']
#rank_metric_list = ['ndcg','ndcg']
rank_metric_list = ['ndcg.1=3.5,0=0.0', 'ndcg.1=3.5,0=0.0']
#rank_metric_list = ['P.10','P.10']
#rank_metric_list = ['infAP','infAP']
#rank_metric_list = ['map', 'bpref']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1] # skiping seed part which is named as 0.1
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}

var = 1

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,8))
for rank_index , rank_metric in enumerate(rank_metric_list):
    for use_ranker in ranker_list:
        for iter_sampling in sampling_list:
            s = ""
            s1 = ""
            originalqrelMap = []
            predictedqrelMap = []
            for datasource in dataset_list:  # 1
                originalqrelMap = []
                predictedqrelMap = []
                originAdress = systemAddress[datasource]
                originalMapResult = originalMapLocation[datasource]


                if preLoaded == False:
                    print "Original Part"
                    fileList = sorted(os.listdir(originAdress))
                    for fileName in fileList:
                        if datasource == 'TREC8' and fileName not in run_name:
                            continue

                        system = originAdress + fileName
                        #shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
                        shellCommand = './trec_eval -m '+rank_metric+ ' ' + qrelAddress[datasource] + ' ' + system

                        shellCommand_start = './trec_eval -m'


                        print shellCommand
                        p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        for line in p.stdout.readlines():
                            print line
                            values = line.split()
                            map = float(values[2])
                            originalqrelMap.append(map)

                        retval = p.wait()

                    originalMapResult = originalMapResult + rank_metric + '_of_all_runs_using_original_qrels.txt'
                    pickle.dump(originalqrelMap, open(originalMapResult, "wb"))

                else:
                    originalMapResult = originalMapResult + rank_metric + '_of_all_runs_using_original_qrels.txt'

                print originalMapResult

                originalqrelMap = pickle.load(open(originalMapResult, "rb"))
                #exit(0)
                print len(originalqrelMap)

                base_address2 = base_address1 + str(datasource) + "/"
                if use_ranker == 'True':
                    base_address3 = base_address2 + "ranker/"
                    s1 = "Ranker and "
                else:
                    base_address3 = base_address2 + "no_ranker/"
                    s1 = "Interactive Search and "
                if iter_sampling == 'True':
                    base_address4 = base_address3 + "oversample/"
                    s1 = s1 + "oversampling"
                else:
                    base_address4 = base_address3 + "no_oversample/"
                    s1 = s1 + "HT correction"

                training_variation = []
                for seed in seed_size:  # 2
                    for batch in batch_size:  # 3
                        for protocol in protocol_list:  # 4
                            print "Dataset", datasource, "Protocol", protocol, "Seed", seed, "Batch", batch
                            s = "Dataset:" + str(datasource) + ", Seed:" + str(seed) + ", Batch:" + str(batch)
                            list = []
                            for fold in xrange(1, 2):
                                predicted_location_base = base_address4 + 'prediction_protocol:' + protocol + '_batch:' + str(
                                    batch) + '_seed:' + str(seed) + '_fold' + str(fold) + '_'
                                print predicted_location_base
                                for percentage in train_per_centage:
                                    predictedqrelMap = []
                                    predictionQrel = ''
                                    predictionModifiedQrel = ''

                                    if rank_index == 1: # 1 bpref human judgement
                                        predictionQrel = predicted_location_base +str(percentage)+'_human_.txt'
                                        predictionModifiedQrel = predicted_location_base +str(percentage)+'_human_modified.txt'
                                    else: # rank_index_ machine + human
                                        predictionQrel = predicted_location_base + str(percentage) + '.txt'
                                        predictionModifiedQrel = predicted_location_base + str(
                                            percentage) + '_modified.txt'

                                    if preLoaded == False:
                                        print predictionQrel
                                        # reading the prediction file and modifyinh it insert one extra column at position 2
                                        f = open(predictionQrel)
                                        tmpstring = ""
                                        for lines in f:
                                            values = lines.split()
                                            tmpstring = tmpstring + values[0] + " 0 " + values[1] + " " + values[2] + "\n"

                                        text_file = open(predictionModifiedQrel, "w")
                                        text_file.write(tmpstring)
                                        text_file.close()

                                        fileList = sorted(os.listdir(originAdress))
                                        for fileName in fileList:
                                            if datasource == 'TREC8' and fileName not in run_name:
                                                continue
                                            system = originAdress + fileName
                                            # shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
                                            shellCommand = './trec_eval -m ' + rank_metric + ' ' + predictionModifiedQrel + ' ' + system

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


                                        #predictionMapResult = predicted_location_base + str(percentage) + '_'+ rank_metric+'.txt'

                                        predictionMapResult = predicted_location_base + str(
                                            percentage) + '_' + rank_metric + '_'+ str(rank_index) +'.txt'

                                        pickle.dump(predictedqrelMap, open(predictionMapResult, 'wb'))
                                    else:
                                        predictionMapResult = predicted_location_base + str(
                                        percentage) + '_' + rank_metric + '_'+ str(rank_index) +'.txt'


                                    predictedqrelMap = pickle.load(open(predictionMapResult, 'rb'))
                                    print len(predictedqrelMap)
                                    tau, p_value = kendalltau(originalqrelMap, predictedqrelMap)
                                    list.append(tau)

                            print list
                            protocol_result[protocol] = list

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
                    if rank_metric == 'infAP':
                        plt.ylabel('tau correlation \n using infAP', size=16)
                    elif rank_metric == 'ndcg.1=3.5,0=0.0':
                        plt.ylabel('tau correlation \n using ndcg', size=16)
                    else:
                        plt.ylabel('tau correlation \n using '+rank_metric_list[0],size = 16)
                if var == 5:
                    if rank_metric == 'infAP':
                        plt.ylabel('tau correlation \n using infAP', size=16)
                    elif rank_metric == 'ndcg.1=3.5,0=0.0':
                        plt.ylabel('tau correlation \n using ndcg', size=16)
                    else:
                        plt.ylabel('tau correlation \n using '+rank_metric_list[1], size=16)

                if var >=5:
                    plt.xlabel('% of human judgments', size=16)

                plt.ylim([0.7, 1])
                plt.yticks([0.7, 0.8, .9, 1.0])
                plt.legend(loc=4)
                if datasource == 'gov2':
                    plt.title('TB\'06', size=16)
                elif datasource == 'WT2013':
                    plt.title('WT\'13', size=16)
                elif datasource == 'WT2014':
                    plt.title('WT\'14', size=16)
                else:
                    plt.title('Adhoc\'99', size=16)

                plt.grid(linestyle='dotted')
                plt.xticks(x_labels_set, x_labels_set)
                var = var + 1

#plt.suptitle(s1, size=16)
plt.tight_layout()
plt.savefig(plotAddress + rank_metric_list[0]+"_"+rank_metric_list[1]+'.pdf', format='pdf')





