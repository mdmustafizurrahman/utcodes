# system packages
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import time
import math

# user packages
from topic_description import TRECTopics


dataset_list = ['WT2014']
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

topicSpecialList = [212, 232, 235, 261, 273, 275, 281, 289, 300, 804]

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

    for topicId in range(start_topic[datasource], end_topic[datasource]):
        print topicId
        if topicId in topicSkipList:
            continue
        if topicId in topicSpecialList:
            continue
        docContents, docLabels = topicAll.get_topic_processed_file(topicId, False)
        rng = np.random.RandomState(31337)
        y = docLabels
        X = docContents
        #kf = KFold(n_splits=numberOfFolds, shuffle=False, random_state=rng)
        f1_xgb = 0.0
        f1_lr = 0.0

        time_xgb = 0
        time_lr = 0

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            stratify=y,
                                                            test_size=0.40)

        #x_train_small, x_test_small, y_train_small, y_test_small = train_test_split(X_train, y_train,
        #                                                    stratify=y,
        #                                                    test_size=0.95)
        length = len(X_train)
        train_end = length
        #train_end = 40
        train_index = list(xrange(0, train_end))
        test_index = list(xrange(train_end, len(X_train)))

        print "LR:",
        start_time = time.time()
        lr_model = LogisticRegression(C=100000000).fit(X_train[train_index], y_train[train_index])
        actuals = y_test
        predictions = lr_model.predict(X_test)
        f1score = f1_score(actuals, predictions, average='binary')
        f1_lr = f1_lr + f1score
        time_spent = time.time() - start_time
        time_lr = time_lr + time_spent
        print f1score, time_spent,

        print "XG:",
        start_time = time.time()
        xgb_model = xgb.XGBClassifier().fit(X_train[train_index], y_train[train_index])
        predictions = xgb_model.predict(X_test)
        #pred_prob = xgb_model.predict_proba(X[test_index])
        #print pred_prob
        f1score = f1_score(actuals, predictions, average='binary')
        f1_xgb = f1_xgb + f1score
        time_spent = time.time() - start_time
        time_xgb = time_xgb + time_spent
        print f1score, time_spent

        f1_per_topic_xgb.append(f1_xgb/float(numberOfFolds))
        f1_per_topic_lr.append(f1_lr/float(numberOfFolds))

        time_per_topic_xgb.append(time_xgb/float(numberOfFolds))
        time_per_topic_lr.append(time_lr/float(numberOfFolds))



    print "total time:", time.time() - initiat_time
    print "Xgb F1 over 50 topics:", np.mean(f1_per_topic_xgb), "time:", np.mean(time_per_topic_xgb)
    print "Lr F1 over 50 topics:", np.mean(f1_per_topic_lr), "time:", np.mean(time_per_topic_lr)