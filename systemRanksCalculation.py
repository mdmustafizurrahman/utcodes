import subprocess
import numpy as np
import sys
import os
import pickle
from systemReader import systemReader
from global_definition import *

os.chdir('/work/04549/mustaf/maverick/data/TREC/trec_eval.9.0')

class systemRanksCalculation:

    def __init__(self, dataset, systemAddress, relevanceJudgementAddress, rankMetric):
        self.dataset = dataset
        self.systemAddress = systemAddress
        self.relevanceJudgementAddress = relevanceJudgementAddress
        self.rankMetric = rankMetric
        self.systemList = None
        self.systemRankMetricValues = {} # key is the systemName and value is the computed RankMetricValue

    # by default all the system under the systemAddress directory are used for systemRank calculation
    # but bysetting this function user can specify a subset of systemList they want to compute
    # systemList is a list of systemName
    def setSystemList(self, systemList):
        self.systemList = systemList

    def calculateSystemRanks(self):
        systemList = sorted(os.listdir(self.systemAddress))
        for systemName in systemList:
            if self.systemList!=None and systemName not in self.systemList:
                continue
            system = self.systemAddress + systemName
            #print self.systemAddress, system
            shellCommand = './trec_eval -m '+self.rankMetric+' ' + self.relevanceJudgementAddress + ' ' + system,
            print shellCommand
            p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                #print line
                values = line.split()
                rankMetricValue = float(values[2])
                print rankMetricValue
                self.systemRankMetricValues[systemName] = rankMetricValue
            retval = p.wait()

        return self.systemRankMetricValues

    def systemRankStats(self, systemId):
        systemRankValues = []
        systemRankerNames = []
        for systemName, systemRankMetricValues in self.systemRankMetricValues.iteritems():
            systemRankValues.append(systemRankMetricValues)
            systemRankerNames.append(systemName)
        print "datasoure:", self.dataset," ",self.rankMetric, " system:", systemId, self.systemRankMetricValues[systemId], " mean", str(np.mean(systemRankValues)), " median", str(np.median(systemRankValues)), " max", str(np.max(systemRankValues)), " std:", str(np.std(systemRankValues))
        print "best run:", systemRankerNames[np.argmax(systemRankValues)], systemRankValues[np.argmax(systemRankValues)]
        print "worst run:", systemRankerNames[np.argmin(systemRankValues)], systemRankValues[np.argmin(systemRankValues)]


rankMetric = 'map'
#dataset_list = ['TREC8', 'gov2','WT2013', 'WT2014']
dataset_list = ['TREC7']

topicSkipList = [202,209,225,237, 245, 255,269, 278, 803, 805] # remember to update the relevance file for this collection accordingly to TAU compute

runList = "input.1 input.8manexT3D1N0 input.acsys8alo input.acsys8amn input.AntHoc1 input.apl8c221 input.apl8n input.att99atdc input.att99atde input.cirtrc82 input.CL99SD input.CL99XT input.disco1 input.Dm8Nbn input.Dm8TFbn input.Flab8as input.Flab8atdn input.fub99a input.fub99tf input.GE8ATDN1 input.ibmg99a input.ibmg99b input.ibms99a input.ibms99b input.ic99dafb input.iit99au1 input.iit99ma1 input.INQ603 input.INQ604 input.isa25 input.isa50 input.kdd8ps16 input.kdd8qe01 input.kuadhoc input.mds08a3 input.mds08a4 input.Mer8Adtd1 input.Mer8Adtd2 input.MITSLStd input.MITSLStdn input.nttd8ale input.nttd8alx input.ok8alx input.ok8amxc input.orcl99man input.pir9Aatd input.pir9Attd input.plt8ah1 input.plt8ah2 input.READWARE input.READWARE2 input.ric8dpx input.ric8tpx input.Sab8A1 input.Sab8A2 input.Scai8Adhoc input.surfahi1 input.surfahi2 input.tno8d3 input.tno8d4 input.umd99a1 input.unc8al32 input.unc8al42 input.UniNET8Lg input.UniNET8St input.UT810 input.UT813 input.uwmt8a1 input.uwmt8a2 input.weaver1 input.weaver2"
rnus_not_in_WT_2014_adhoc = ['UDInfoWebRiskTR','UDInfoWebRiskRM','UDInfoWebRiskAX','ICTNET14RSR1','ICTNET14RSR2','uogTrq1','uogTrBwf','ICTNET14RSR3','udelCombCAT2','uogTrDwsts','wistud.runD','wistud.runE']
run_name = []
for run in runList.split(" "):
    run_name.append(run)

def calculateNumberofRelevantDocuments(qrelDocumentsForTopic):
    n_relevant_docs = 0
    for docNo, docLabel in qrelDocumentsForTopic.iteritems():
        if docLabel == 1:
            n_relevant_docs = n_relevant_docs + 1
    return n_relevant_docs


def calculateMissRate(systemRankedDocuments,qrelDocuments,missRelevantDocs=True):
    missRate = []
    number_of_ranked_documents_per_topic = []
    for topicId in sorted(systemRankedDocuments.iterkeys()):
        if int(topicId) in topicSkipList:
            continue
        # qrelDocumentsForTopic is a dictionary with key-> docNo and value-> label
        qrelDocumentsForTopic = qrelDocuments[topicId]
        count = 0
        if missRelevantDocs == True: # only care about how much the ranker miss the releavnt documnet
            n_relevant_docs = calculateNumberofRelevantDocuments(qrelDocumentsForTopic)
            for docNo in qrelDocumentsForTopic.iterkeys():
                # systemRankedDocuments[topicId] is a dictionary with key-> docNo and value-> rank
                if docNo not in systemRankedDocuments[topicId] and qrelDocumentsForTopic[docNo] == 1:
                    count = count + 1
            missRate.append(float(count)/float(n_relevant_docs))
        else:
            for docNo in qrelDocumentsForTopic.iterkeys():
                # systemRankedDocuments[topicId] is a dictionary with key-> docNo and value-> rank
                if docNo not in systemRankedDocuments[topicId]:
                    count = count + 1

            missRate.append(float(count)/float(len(qrelDocumentsForTopic)))

        #print topicId, len(systemRankedDocuments[topicId]), len(qrelDocumentsForTopic), count
        number_of_ranked_documents_per_topic.append(len(systemRankedDocuments[topicId]))
    return np.mean(missRate)*100, np.mean(number_of_ranked_documents_per_topic)





for datasource in dataset_list:
    systemRunsValues = systemRanksCalculation(datasource, systemAddress[datasource], qrelAddress[datasource], rankMetric)
    if datasource == 'TREC8':
        systemRunsValues.setSystemList(run_name)
    systemRunsValues.calculateSystemRanks()
    systemRunsValues.systemRankStats(systemName[datasource])

exit(0)
for datasource in dataset_list:
    systemRankedDocumentObject = systemReader(datasource, start_topic[datasource], end_topic[datasource])
    systemRankedDocuments = systemRankedDocumentObject.rankedDocumentFromSystem(systemAddress[datasource],systemName[datasource])
    #bestSystemRankedDocuments = systemRankedDocumentObject.rankedDocumentFromSystem(bestSystem[datasource])

    qrelDocuments = systemRankedDocumentObject.qrelsReader()

    missrate = {}
    missrate['random'] = calculateMissRate(systemRankedDocuments,qrelDocuments)
    missrate['best']   = calculateMissRate(bestSystemRankedDocuments,qrelDocuments)
    print datasource
    for systemType, systemMissRate in missrate.iteritems():
        print systemType, systemMissRate[0],systemMissRate[1]
