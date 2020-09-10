import numpy as np
import sys
import os
import pickle
from collections import OrderedDict

class systemReader:
    def __init__(self, dataset, startTopic, endTopic):
        self.dataset = dataset
        self.startTopic = startTopic # integer values
        self.endTopic = endTopic # integer values # given end topic is always greater than 1


    # return a dictionary
    # where key is the topicid (str)
    # values is a dictionary of (documentNo, rank from TREC)
    def rankedDocumentFromSystem(self, systemAddress, systemName, pool_depth):
        systemNameAddress = systemAddress + systemName
        f = open(systemNameAddress)
        topicInfo = {}  # key--TopicID values -- Dictionary(rank, docNo)
        for lines in f:
            values = lines.split("\t")
            topicNo = values[0]
            if int(topicNo) < self.startTopic or int(topicNo) >= self.endTopic:
                continue
            docNo = values[2]
            docNo =docNo.strip()
            rank_no = int(values[3])
            if rank_no > pool_depth:
                continue
            if topicNo in topicInfo:
                docNo_to_rank = topicInfo[topicNo]
                docNo_to_rank[docNo] = rank_no
                topicInfo[topicNo] = docNo_to_rank
            else:
                docNo_to_rank = {}
                docNo_to_rank[docNo] = rank_no
                topicInfo[topicNo] = docNo_to_rank

        f.close()
        # testing purpose
        # for topicId in sorted(topicInfo.iterkeys()):
        #    print topicId, len(topicInfo[topicId])

        return topicInfo

    def minimunNumberofQueryVariants(self, topicInfo):
        minNumberOfVariants = 1000  # a max number is put
        minTopic = 0
        maxNumberOfVariants = 0
        for topicNo, queryInfo in sorted(topicInfo.iteritems()):
            #print topicNo, len(queryInfo.keys())
            if len(queryInfo.keys()) <= minNumberOfVariants:
                minNumberOfVariants = len(queryInfo.keys())
                minTopic = topicNo
            maxNumberOfVariants = max(maxNumberOfVariants, len(queryInfo.keys()))

        return minNumberOfVariants, maxNumberOfVariants

    def popularQueryVariantsCountedByNumberOfRelevantDocuments(self, topicInfo, topicQrels, data_path, data_source, systemName):

        file_complete_path = data_path + data_source + "_" + systemName + "_popular_query_by_relevant_docs" + '.pickle'
        print "Popular Query by Relevant Doc:", file_complete_path
        topicQueryInfo = {} # topicNo is key, queryInfo values,
        # queryProductivityInfo = {} queryVariants is key, numberofrelevantdoc is values

        for topicNo, queryInfo in sorted(topicInfo.iteritems()):
            queryProductivityInfo = {}
            for queryVariant, docList in sorted(queryInfo.iteritems()):
                relevantDocumentCounts = 0
                for docNo in docList:
                    # getting labels from qrels
                    # not all ranked document in qrels
                    if docNo in topicQrels[str(topicNo)]:
                        label = topicQrels[str(topicNo)][docNo]
                        if label == 1:
                            relevantDocumentCounts = relevantDocumentCounts + 1

                if topicNo in topicQueryInfo:
                    queryProductivityInfo = topicQueryInfo[topicNo]

                queryProductivityInfo[queryVariant] = relevantDocumentCounts

            queryProductivityInfoSorted = OrderedDict(sorted(queryProductivityInfo.items(), key=lambda x: (-1)*x[1]))

            topicQueryInfo[topicNo] = queryProductivityInfoSorted


        topicQueryInfoList = {}
        for topicNo, queryProductivityInfo in sorted(topicQueryInfo.iteritems()):
            queryList = [] # queryList is a list of queryVariant order by their popularity count
            for queryVariant, relevantCount in queryProductivityInfo.iteritems():
                print topicNo, queryVariant, relevantCount
                queryList.append(queryVariant)
            topicQueryInfoList[topicNo] = queryList

        pickle.dump(topicQueryInfoList, open(file_complete_path, 'wb'))
        return topicQueryInfoList

    # UQV files contains both WT2013 and WT2014
    # so we need to transfer the datasource also
    def documentListFromUQVVariants(self, topic_qrels, systemAddress, systemName, data_source, data_path):
        file_complete_path = data_path + data_source + "_" + systemName + '.pickle'
        print "UQV--system-run-file:", file_complete_path
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            topicInfo = pickle.load(open(file_complete_path, 'rb'))
        else:
            print file_complete_path + " does not exists. Creating " + file_complete_path
            systemNameAddress = systemAddress + systemName
            f = open(systemNameAddress)
            topicInfo = {}  # key--TopicID values -- Dictionary(rank, docNo)
            topicNo = -1
            queryVariants = ""
            docNo = ""
            docList = []
            for lines in f:
                if "# topic = " in lines:
                    #print lines
                    topicNo = int(lines[lines.find("=")+2:])
                    #print topicNo

                if "# query = " in lines:
                    #print lines
                    queryVariants = lines[lines.find("=") + 2:].strip()
                    #print queryVariants
                if "#" not in lines:
                    #print lines
                    docNo = lines.split(" ")[1].strip()
                    if str(topicNo) in topic_qrels:
                        if docNo in topic_qrels[str(topicNo)]:
                            docList.append(docNo)
                            docScore = lines.split(" ")[2].strip()
                            #print docNo, docScore

                if "##############################################" in lines:
                    if int(topicNo) < self.startTopic or int(topicNo) >= self.endTopic:
                        continue

                    queryInfo = {}
                    if topicNo in topicInfo:
                        queryInfo = topicInfo[topicNo]

                    queryInfo[queryVariants] = docList
                    topicInfo[topicNo] = queryInfo

                    docList = []
                    topicNo = -1
                    queryVariants = ""
                    docNo = ""

            f.close()
            pickle.dump(topicInfo, open(file_complete_path, 'wb'))


            # code for sanity checking
            '''
            for topicNo, queryInfo in sorted(topicInfo.iteritems()):
                queryVariantsNo = 0
                for queryVariants, docList in sorted(queryInfo.iteritems()):
                    print topicNo, queryVariants, len(docList)
                    for docNo in docList:
                        print docNo
                    if queryVariantsNo > 1:
                        break
                    queryVariantsNo = queryVariantsNo + 1

                break
            '''
        return topicInfo

    # topicInfo is dictionary where topicId is key
    # values is a dictionry(QueryVariants, DocList)
    def getUQVPooledBudgetAndRelevantCount(self, topicInfo, topic_qrels, pool_depth, data_path, data_source, systemName):

        file_name = data_path + data_source + "_" + systemName + "_topicPooledBudgetRelCountInfo" + '.pickle'
        topicPoolInfo = {}  # pool number is the key, value is pool_depth --> [DocCount, Relevant Doc]

        if os.path.exists(file_name):
            print "topic pool info file exits at:", file_name
            topicPoolInfo = pickle.load(open(file_name, "rb"))
            return topicPoolInfo
        else:
            print "topic Pool Info Does not Exists at. Creating file", file_name
            for topicId, queryInfo in sorted(topicInfo.iteritems()):
                all_doc_list = [] # will hold list of documents at each pool depth
                relevant_count = 0
                poolInfo = {}
                for pool in xrange(0, pool_depth):
                    for queryVariants, docList in sorted(queryInfo.iteritems()):
                        docNo = docList[pool]
                        if docNo in topic_qrels[str(topicId)]:
                            if docNo not in all_doc_list:
                                all_doc_list.append(docNo)
                                docLabel = topic_qrels[str(topicId)][docNo]
                                relevant_count = relevant_count + docLabel
                    poolInfo[pool] = [len(all_doc_list), relevant_count]

                topicPoolInfo[topicId] = poolInfo

            pickle.dump(topicPoolInfo, open(file_name, "wb"))
            return topicPoolInfo
            # sanity check
            '''
            for topicId, poolInfo in sorted(topicPoolInfo.iteritems()):
                for pool, infoList in sorted(poolInfo.iteritems()):
                    print topicId, pool, infoList
            '''



    # return a dictionary
    # where key is the topicid (str)
    # values is a dictionary of (documentNo, rank from TREC)
    def documentListFromSystem(self, systemAddress, systemName, data_path):

        file_complete_path = data_path + systemName + '.pickle'
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            topicInfo = pickle.load(open(file_complete_path, 'rb'))
        else:
            print file_complete_path + " does not exists. Creating " + file_complete_path
            systemNameAddress = systemAddress + systemName
            f = open(systemNameAddress)
            topicInfo = {}  # key--TopicID values -- Dictionary(rank, docNo)
            for lines in f:
                values = lines.split("\t")
                topicNo = values[0]
                if int(topicNo) < self.startTopic or int(topicNo) >= self.endTopic:
                    continue
                docNo = values[2]
                docNo = docNo.strip()
                if topicNo in topicInfo:
                    doc_list = topicInfo[topicNo]
                    doc_list.append(docNo)
                    topicInfo[topicNo] = doc_list
                else:
                    doc_list = []
                    doc_list.append(docNo)
                    topicInfo[topicNo] = doc_list

            f.close()
            pickle.dump(topicInfo, open(file_complete_path,'wb'))
        return topicInfo

