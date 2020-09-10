# system packages
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
#from yellowbrick.classifier import DiscriminationThreshold
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# user packages
from topic_description import TRECTopics


dataset_list = ['TREC8']
qrelAddress = {}
qrelAddress['TREC8'] = '/work/04549/mustaf/maverick/data/TREC/TREC8/relevance.txt'
qrelAddress['gov2'] = '/work/04549/mustaf/maverick/data/TREC/gov2/qrels.tb06.top50.txt'
qrelAddress['WT2013'] = '/work/04549/mustaf/maverick/data/TREC/WT2013/modified_qreldocs2013.txt'
qrelAddress['WT2014'] = '/work/04549/mustaf/maverick/data/TREC/WT2014/modified_qreldocs2014.txt'

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

bestSystem = {}
bestSystem['TREC8'] = 'input.READWARE2'
bestSystem['gov2'] = 'input.indri06AtdnD'
bestSystem['WT2013'] = 'input.clustmrfaf'
bestSystem['WT2014'] = 'input.uogTrDwl'

start_topic = {}
start_topic['TREC8'] = 401
start_topic['gov2'] = 801
start_topic['WT2013'] = 201
start_topic['WT2014'] = 251

end_topic = {}
end_topic['TREC8'] = 451
end_topic['gov2'] = 851
end_topic['WT2013'] = 251
end_topic['WT2014'] = 301

pooledProcessedDocumentsPath = {}

pooledProcessedDocumentsPath['TREC8'] = "/work/04549/mustaf/maverick/data/TREC/TREC8/processed.txt"
pooledProcessedDocumentsPath['gov2'] = "/work/04549/mustaf/maverick/data/TREC/gov2/processed.txt"
pooledProcessedDocumentsPath['WT2013'] = "/work/04549/mustaf/maverick/data/TREC/WT2013/processed_new.txt"
pooledProcessedDocumentsPath['WT2014'] = "/work/04549/mustaf/maverick/data/TREC/WT2014/processed_new.txt"

topicSkipList = [202,209,225,237, 245,255,269,278, 803, 805] # remember to update the relevance file for this collection accordingly to TAU compute
numberOfFolds = 1

#topicSpecialList = [212, 232, 235, 261, 273, 275, 281, 289, 300, 804]
line_color = ['-b', '-g', '-r', '-c', '-m', '-y']

for datasource in dataset_list:
    topicAll = TRECTopics(datasource, start_topic[datasource], end_topic[datasource])
    topicAll.set_pooled_processed_document_path(pooledProcessedDocumentsPath[datasource])
    topicAll.load_prcessed_document()
    topicAll.load_relevance_judgements(qrelAddress[datasource])

    f1_per_topic_xgb = []
    f1_per_topic_lr = []

    time_per_topic_xgb = []
    time_per_topic_lr = []

    initiat_time = time.time()
    relevance_ratio_list = []
    topicIdToTopicIndex = {}
    topicIdToTopicIndexCounter = 0
    '''
    for topicId in range(start_topic[datasource], end_topic[datasource]):
        print topicId
        if topicId in topicSkipList:
            continue
        docContents, docLabels = topicAll.get_topic_processed_file(topicId, False)
        rng = np.random.RandomState(31337)
        y = docLabels
        X = docContents
        relevance_ratio = float(np.count_nonzero(y))/float(len(y))
        relevance_ratio_list.append(relevance_ratio)
        topicIdToTopicIndex[topicIdToTopicIndexCounter] = topicId
        topicIdToTopicIndexCounter = topicIdToTopicIndexCounter + 1

    minRelevanceTopicId = topicIdToTopicIndex[np.argmin(relevance_ratio)]
    maxRelevanceTopicId = topicIdToTopicIndex[np.argmax(relevance_ratio)]

    f1_lr = 0.0

    time_lr = 0
    # loading the lowest relevancetopic content
    print "Lowest Prevalence Topic:", minRelevanceTopicId
    print "Highest Prevalence Topic:", maxRelevanceTopicId
    '''
    docContents, docLabels = topicAll.get_topic_processed_file('402', False)
    y = docLabels
    X = docContents
    rng = np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.40)

    length = len(X_train)
    train_end = int(math.ceil(len(X)*0.3))
    #train_end = 40
    train_index = list(xrange(0, train_end))
    test_index = list(xrange(train_end, len(X_train)))

    print "LR:",
    start_time = time.time()

    ros = RandomOverSampler()
    X_train_oversampled, y_train_oversampled = ros.fit_sample(X_train[train_index], y_train[train_index])
    lr_model_oversampled = LogisticRegression(C=100000000).fit(X_train_oversampled, y_train_oversampled)
    lr_model = LogisticRegression(C=100000000).fit(X_train[train_index], y_train[train_index])
    lr_model_default = LogisticRegression().fit(X_train[train_index], y_train[train_index])

    cutt_off_list = [0.5, 0.6, 0.7, 0.8, 0.9]

    plt.plot([0, 1], [0, 1], 'k--')
    y_pred_prob = lr_model.predict_proba(X_test)
    print y_pred_prob
    print np.sum(y_pred_prob[0])
    print y_pred_prob[0][0], y_pred_prob[0][1], y_pred_prob[0][0] + y_pred_prob[0][1]
    exit(0)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    print tpr, fpr, thresholds
    plt.plot(fpr, tpr, line_color[0],
             label= 'LR (C = 10^8) AUC:' + str(roc_auc_score(y_test, y_pred_prob))[:4])

    y_pred_prob = lr_model_oversampled.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob,pos_label=1)
    print tpr, fpr, thresholds
    plt.plot(fpr, tpr, line_color[1],
             label='OS LR (C = 10^8) AUC:' + str(roc_auc_score(y_test, y_pred_prob))[:4])



    y_pred_prob = lr_model_default.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    print tpr, fpr, thresholds
    plt.plot(fpr, tpr, line_color[2],
             label='LR (C = 1) AUC:' + str(roc_auc_score(y_test, y_pred_prob))[:4])

    '''
    for i, cutt_off in enumerate(cutt_off_list):
        y_pred = (lr_model.predict_proba(X_test)[:, 1] >= cutt_off).astype(bool)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        print cutt_off, thresholds
        plt.plot(fpr, tpr, line_color[i+1], label='cutt off:'+str(cutt_off)+'AUC:'+str(roc_auc_score(y_test, y_pred))[:4])
    '''
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    #plt.show()
    plt.savefig('Cutt_off_default_os.pdf', format='pdf')

'''
    plt.close()

    logistic = LogisticRegression()
    visualizer = DiscriminationThreshold(logistic)
    visualizer.fit(X_train_oversampled, y_train_oversampled)  # Fit the training data to the visualizer
    visualizer.poof()
'''

'''
pred_proba_df = pd.DataFrame(model.predict_proba(x_test))
threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
for i in threshold_list:
    print ('\n******** For i = {} ******'.format(i))
    Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
    test_accuracy = metrics.accuracy_score(Y_test.as_matrix().reshape(Y_test.as_matrix().size,1),
                                           Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))
    print('Our testing accuracy is {}'.format(test_accuracy))

    print(confusion_matrix(Y_test.as_matrix().reshape(Y_test.as_matrix().size,1),
                           Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1)))
'''