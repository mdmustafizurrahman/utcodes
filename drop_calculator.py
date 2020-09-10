from scipy.stats.stats import kendalltau
from numpy import trapz
import os
import matplotlib
import operator
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#gs = gridspec.GridSpec(5, 2)

os.chdir('/work/04549/mustaf/maverick/data/TREC/trec_eval.9.0')

#base_address1 = "/home/nahid/UT_research/clueweb12/bpref_result/"
#plotAddress = "/home/nahid/UT_research/clueweb12/bpref_result/plots/tau/mapbpref/"
#baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"


base_address1 = "/work/04549/mustaf/maverick/data/TREC/"
plotAddress =  "/work/04549/mustaf/maverick/data/TREC/plot/"

ht_estimation = False
crowd = True

dataset_list = ['TREC8']
protocol_list = ['CAL']


#dataset_list = ['TREC8']

ranker_list = ['True']
sampling_list = ['True']
map_list = ['map']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # skiping seed part which is named as 0.1
#train_per_centage = [0.1]
#train_per_centage = [0.2, 0.3] # skiping seed part which is named as 0.1
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#x_labels_set =[10,20]



result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
#subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]
#subplot_loc = [221,222,223,224]

budget_list = []
budget_list_TREC8 = [10134, 19231, 28125, 36947, 45584, 54071, 62476, 70654, 78805]
import pickle
'''
for budget in xrange(2000, last_budget, budget_increment):
    budget_list.append(budget)
    budget_limit_to_train_percentage_mapper[budget] = budget
'''
datasource = 'TREC8'
if datasource == 'TREC8':
    for budget in budget_list_TREC8:
        budget_list.append(budget)

budget_list = sorted(budget_list)
import operator

# List1 is the ground truth and list 2 is the predicted list
def tau_ap_mine(list1, list2):

    length = len(list2)
    c = [0] * length

    for i in xrange(1, len(list2)):
        index_of_element_in_i_list2_in_ground_list = list1.index(list2[i])
        for j in xrange(0,i):
            index_of_element_j_in_list2_in_ground_list = list1.index(list2[j])
            if index_of_element_in_i_list2_in_ground_list > index_of_element_j_in_list2_in_ground_list:
                c[i-1] += 1

    summation = 0
    for i in xrange(1,length):
        summation = summation + (1.0 * c[i-1] / (i))

    p = float(summation)/ (length-1)

    return 2 * p - 1.0

# drop calculator
def drop_calculator(original_list, predicted_list):
    # create a dictionary from original_list and predicted list
    # key is the index of systems
    original_Map = {}
    predicTed_Map = {}

    for i, value in enumerate(original_list):
        original_Map[i] = value

    i = 0
    for i, value in enumerate(predicted_list):
        predicTed_Map[i] = value

    # sort both dictionary by value
    # output is a list of tuple (indexofsystem, value)
    sorted_original_Map = sorted(original_Map.items(), key=operator.itemgetter(1))

    sorted_predicted_Map = sorted(predicTed_Map.items(), key=operator.itemgetter(1))


    original_rank_list = range(len(sorted_original_Map))
    predicted_rank_list = []

    max_diff = 0
    for rank_in_original_list, x in enumerate(sorted_original_Map):
        # find that system in predicted list along with it ranks
            for rank_in_predicted_list, y in enumerate(sorted_predicted_Map):
                if y[0] == x[0]: # is system id match
                    predicted_rank_list.append(rank_in_predicted_list)
                    if abs(rank_in_predicted_list - rank_in_original_list) > max_diff:
                        max_diff = abs(rank_in_predicted_list - rank_in_original_list)
                    break

    print predicted_rank_list

    return max_diff, tau_ap_mine(original_rank_list, predicted_rank_list)





var = 1
stringUse = ''
#fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,6))
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

                    qrelAdress = base_address1 + datasource + '/relevance.txt'
                    originalMapResultBase = base_address1 + datasource + "/"

                    predictionAddress = base_address1 +'/deterministic1/TREC8/result/ranker/oversample/MAB/5/'
                    predictionModifiedAddress = base_address1 +'/deterministic1/TREC8/result/ranker/oversample/MAB/5'
                    lambda_param = 0.75
                    alpha_param = 1
                    train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                         1.0, ]  # skiping seed part which is named as 0.1

                elif datasource == 'WT2013':
                    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modified_qreldocs2013.txt'
                    originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/'
                    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/prediction/"
                    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modifiedprediction/"
                    lambda_param = 0.75
                    alpha_param = 2
                    train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                         1.0, ]  # skiping seed part which is named as 0.1


                else:
                    originAdress = base_address1 + datasource + "/systemRankings/"
                    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modified_qreldocs2014.txt'
                    originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/'
                    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/prediction/"
                    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modifiedprediction/"
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

                originalMapResult = originalMapResultBase + stringUse + '_of_all_runs_using_original_qrels.txt'
                originalqrelMap = pickle.load(open(originalMapResult, "rb"))

                tau_list = {} # budget is the key, value is the tau
                ##############################

                for budget in budget_list:
                    predictionMapResult = predictionAddress + "prediction_protocol:CAL_batch:1_seed:10_fold1_"+str(budget)+"_human_"+stringUse+".pickle"

                    predictedqrelMap = pickle.load(open(predictionMapResult, "rb"))

                    tau, p_value = kendalltau(originalqrelMap, predictedqrelMap)
                    tau_list[budget] = tau

                    print budget, tau, drop_calculator(originalqrelMap, predictedqrelMap)

                    predictedqrelMap = []  # cleaning it for next trains_percenatge

                for key in sorted(tau_list.iterkeys()):
                    print key, tau_list[key]

                predictiontauResult = predictionAddress + "prediction_protocol:CAL_batch:1_seed:10_fold1_budget_all_human_tau_on_"+stringUse+".pickle"

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





