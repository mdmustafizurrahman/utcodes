import sys
from lxml import etree
import codecs
from bs4 import BeautifulSoup
import os
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool as ProcessPool
import itertools
from functools import partial
import time
from tqdm import tqdm

#topicId or topicNo is the id from TREC like 401, 851
#topicIndex is for my internal perpuse always started with 0 index
class TRECTopics:

    def __init__(self,dataset,startTopic, endTopic):
        self.dataset = dataset
        self.startTopic = startTopic
        self.endTopic = endTopic
        self.topic_to_doclist = {}  # key is the topic(string) and value is the list of docNumber
        self.docId_label = {}  # key is the DocNo and the value is the label
        self.topicDescription = {} # key is the topic id in string and value is the string descritption

        self.pooledProcessedDocumentPath = None
        self.topicDescriptionFilePath = None


    def rawDocuments_to_processed(self,raw_document):
        letters_only = re.sub("[^a-zA-Z]", " ", raw_document)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        return (" ".join(meaningful_words))

    def set_pooled_processed_document_path(self, pooledProcessedDocumentsPath):
        self.pooledProcessedDocumentPath =  pooledProcessedDocumentsPath

    def set_topic_description_path(self, topicDescriptionFilePath):
        self.topicDescriptionFilePath = topicDescriptionFilePath

    def process_topic_description(self):
        soup = BeautifulSoup(codecs.open(self.topicDescriptionFilePath, "r"), "lxml")
        topicId = self.startTopic
        for topic in soup.findAll("top"):
            description = str(topic.find('desc').text).replace("Description:","").replace("Narrative:","")
            #description = ' '.join(description.split())
            self.topicDescription[str(topicId)] = self.rawDocuments_to_processed(description)
            topicId = topicId + 1

    # porcessed Pooled documents are all the documents in the pools for all topics
    # this is a dictionary where key is the documnet id from TREC and value is the processed text
    def load_prcessed_document(self):
        self.processedPooledDocuments = pickle.load(open(self.pooledProcessedDocumentPath,'rb'))

    # read the qrel and return a list of docId in the original qrel file
    def qrelDocIdLister(self,qrelsAddress, data_path, file_name):
        docList = []  # list of docId from qrels
        # print qrelsAddress
        file_complete_path = data_path + file_name + '.pickle'
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            docList = pickle.load(open(file_complete_path, 'rb'))
        else:
            f = open(qrelsAddress)
            for lines in f:
                values = lines.split()
                topicNo = values[0]
                # print values
                if int(topicNo) < self.startTopic or int(topicNo) >= self.endTopic:
                    continue
                docId = values[2]
                docId = docId.strip()
                if docId not in docList:
                    docList.append(docId)
            f.close()
            pickle.dump(sorted(docList), open(file_complete_path, 'wb'))
        return docList

    # load relevance judgement per topic wise
    # return a dictionary of topic_to_docList key is topicid from TREC and doclist is the list of document related to that topic
    def qrelsReader(self, qrelsAddress, data_path, file_name):
        topicInfo = {}  # key--TopicID str values -- Dictionary(docNo, label) # label is integer
        #print qrelsAddress
        file_complete_path = data_path+file_name+'.pickle'
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            topicInfo = pickle.load(open(file_complete_path, 'rb'))
        else:
            print file_complete_path + " does not exists. Creating " + file_name
            f = open(qrelsAddress)
            for lines in f:
                values = lines.split()
                topicNo = values[0]
                #print values
                if int(topicNo) < self.startTopic or int(topicNo) >= self.endTopic:
                    continue
                docNo = values[2]
                docNo = docNo.strip()
                label = int(values[3])

                if label > 1:
                    label = 1
                if label < 0:
                    label = 0

                if topicNo in topicInfo:
                    docNo_label = topicInfo[topicNo]
                    docNo_label[docNo] = label
                    topicInfo[topicNo] = docNo_label
                else:
                    docNo_label = {}
                    docNo_label[docNo] = label
                    topicInfo[topicNo] = docNo_label

            f.close()
            pickle.dump(topicInfo,open(file_complete_path,'wb'))
        return topicInfo


    # load relevance judgement per topic wise
    # return a dictionary of topic_to_docList key is topicid from TREC and doclist is the list of document related to that topic
    def topicBudgetFromOfficialQrels(self, qrelsAddress, data_path, file_name):
        topicInfo = {}
        topicBudget  = {}  # key--TopicID str values -- Dictionary(docNo, label) # label is integer
        # print qrelsAddress
        file_complete_path = data_path + file_name + '.pickle'
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            topicBudget = pickle.load(open(file_complete_path, 'rb'))
            return topicBudget
        else:
            print file_complete_path + " does not exists. Creating " + file_name
            f = open(qrelsAddress)
            for lines in f:
                values = lines.split()
                topicNo = values[0]
                # print values
                if int(topicNo) < self.startTopic or int(topicNo) >= self.endTopic:
                    continue
                docNo = values[2]
                docNo = docNo.strip()
                label = int(values[3])
                if label > 1:
                    label = 1
                if label < 0:
                    label = 0

                if topicNo in topicInfo:
                    docNo_label = topicInfo[topicNo]
                    docNo_label[docNo] = label
                    topicInfo[topicNo] = docNo_label
                else:
                    docNo_label = {}
                    docNo_label[docNo] = label
                    topicInfo[topicNo] = docNo_label

            f.close()
            #pickle.dump(topicInfo, open(file_complete_path, 'wb'))

        for topicId, docNo_label_dic in topicInfo.iteritems():
            #number_of_relevants = 0
            #for docNo, docLabel in docNo_label_dic.iteritems():
            #    if docLabel == 1:
            #        number_of_relevants = number_of_relevants + 1
            topicBudget[topicId] = len(docNo_label_dic)
        pickle.dump(topicBudget, open(file_complete_path, 'wb'))

        return topicBudget

    def construct_original_qrels(self, docIdToDocIndex, topic_qrels, data_path, file_name):
        original_topic_qrels = {}
        file_complete_path = data_path + file_name + ".pickle"
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            original_topic_qrels = pickle.load(open(file_complete_path, 'rb'))
        else:
            print file_complete_path + " does not exists. Creating " + file_name
            with tqdm(total=len(topic_qrels)) as pbar:
                for topicId in sorted(topic_qrels.iterkeys()):
                    # document_to_label is a dictionary of document related to topicId
                    document_to_label = topic_qrels[topicId]
                    original_labels = {}  # key is the documentIndex, values is the label
                    for document_id in sorted(document_to_label.iterkeys()):
                        document_label = document_to_label[document_id]
                        document_index = docIdToDocIndex[document_id]
                        original_labels[document_index] = document_label
                    original_topic_qrels[topicId] = original_labels
                    pbar.update()
                pickle.dump(original_topic_qrels, open(file_complete_path, 'wb'))

        return original_topic_qrels

    # total 7 parametes
    # classifier,classifier_name, document_collection,
    # docIdToDocIndex, topic_qrels, data_path, file_name):

    # use per topic original qrels to induce a classifier and
    # use that classifier to predict the relevance
    # of all the documents in the collection
    # classifier --> the type of the classifier
    # document_collection --> a collection of documents in CSR format with TF-IDF or any representation, indexed by documentIndex
    # docIdToDocIndex is a dictionary from documentID to document index in the document collection
    # topic qrels contains the original topic wise related document list and labels (topicId is the key value is the list of document)
    # data_path --> path to find and write everything
    # fileName -->

    # retunrs a predicted_topic_qrels which is a tuple of threes element(original_labels, predicted_labels, complete_labels))
    # where original_labels and predicted_labels are a dictionary --> key, document_index, values-- lables
    # complete_labels is a list of labels combined from original and predicted
    # this complete_labels is a list indexd from 0 to total_document in document collection
    # This is a serial version and extremely
    # for parallel version using multi=processing
    # we need place the function under __main__ global script
    # otherwise multi does not worl
    # see tfIdfLoader.py to check
    def construct_predicted_qrels(self, classifier,classifier_name, document_collection, docIdToDocIndex, topic_qrels, data_path, file_name):
        predicted_topic_qrels = {}
        file_complete_path = data_path + file_name + "_"+classifier_name+".pickle"
        total_documents = len(docIdToDocIndex)
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            predicted_topic_qrels = pickle.load(open(file_complete_path, 'rb'))
        else:

            with tqdm(total=len(topic_qrels)) as pbar:
                for topicId in sorted(topic_qrels.iterkeys()):

                    #document_to_label is a dictionary of document related to topicId
                    document_to_label = topic_qrels[topicId]
                    train_index_list = []
                    train_labels = []
                    original_labels = {} # key is the documentIndex, values is the label
                    predicted_labels = {} # key is the documentIndex, values is the label
                    complete_labels = [] # indexed by the documentIndex, sameorder from 0 to totalDocuments
                    for document_id, document_label in document_to_label.iteritems():
                        document_index = docIdToDocIndex[document_id]
                        train_index_list.append(document_index)
                        train_labels.append(document_label)
                        original_labels[document_index] = document_label

                    X = document_collection[train_index_list]
                    y = train_labels
                    model = None
                    if classifier!= None:
                        model = classifier.fit(X,y)
                    for document_index in xrange(0,total_documents):
                        if document_index not in train_index_list:
                            if classifier!=None:
                                predicted_labels[document_index] = model.predict(document_collection[document_index])
                                complete_labels.append(predicted_labels[document_index])
                            else:
                                complete_labels.append(0) # not relevant any document outside pool
                        else:
                            complete_labels.append(original_labels[document_index])

                    predicted_topic_qrels[topicId] = (original_labels, predicted_labels, complete_labels)
                    pbar.update()
                pickle.dump(predicted_topic_qrels,open(file_complete_path,'wb'))

            return predicted_topic_qrels
    # topic_qrels is a dictionary where key is the topicId and values is the document_to_label
    # document_to_label is a dictionary key is the documentID from TRCE and value is the label
    # topic_complete_qrels_address is the base_address
    # to get a topic specific file we need to append the topicID
    # topic_complete_qrels = pickle.load(open(topic_complete_qrels_address + topicId + '.pickle', 'rb'))
    # this loads pickle file which format is a tuple of (original_labels, predicted_labels)
    # where original and predicted qrel
    # is a dictionary where value is a documentIndex from document collection
    # number of seed should be an EVEN number
    # value is the label
    '''
    def get_topic_seed_documents(self, topic_qrels, topic_complete_qrels_address, number_of_seeds, seed_selection_type, data_path, file_name):
        topic_seed_info = {}
        file_complete_path = data_path + file_name + '.pickle'
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            topic_seed_info = pickle.load(open(file_complete_path, 'rb'))
        else:
            # interactive search for seed selection
            print "File not existed, creating seed list using "+ seed_selection_type
            if seed_selection_type == 'IS':
                with tqdm(total=len(topic_qrels)) as pbar:
                    for topicId in sorted(topic_qrels.iterkeys()):
                        # topic_complete_qrels is a tuple of  (original_labels, predicted_labels)
                        topic_complete_qrels = pickle.load(open(topic_complete_qrels_address + topicId + '.pickle', 'rb'))
                        #print topicId
                        original_qrel = topic_complete_qrels[0] # 0 is the original_labels
                        train_index_list = []
                        seed_one_counter = 0.0
                        seed_zero_counter = 0.0
                        for document_index in sorted(original_qrel.iterkeys()):
                            if seed_zero_counter == number_of_seeds/2.0 and seed_one_counter == number_of_seeds/2.0:
                                break
                            doc_label = original_qrel[document_index]
                            if doc_label == 1 and seed_one_counter<(number_of_seeds/2.0):
                                train_index_list.append(document_index)
                                seed_one_counter = seed_one_counter + 1.0
                            elif doc_label == 0 and seed_zero_counter<(number_of_seeds/2.0):
                                train_index_list.append(document_index)
                                seed_zero_counter = seed_zero_counter + 1.0
                        topic_seed_info[topicId] = train_index_list
                        pbar.update()
                pickle.dump(topic_seed_info, open(file_complete_path, 'wb'))

        return topic_seed_info

    '''

    def get_topic_seed_documents(self, topic_qrels, topic_original_qrels_in_doc_index, docIdToDocIndex, number_of_seeds, seed_selection_type, data_path, file_name, documentIdListFromRanker=None):
        topic_seed_info = {}
        file_complete_path = data_path + file_name + '.pickle'
        # if file already exist, just load it
        if os.path.isfile(file_complete_path):
            print file_complete_path + " exists. Loading from that."
            topic_seed_info = pickle.load(open(file_complete_path, 'rb'))
        else:
            # interactive search for seed selection
            print file_complete_path + " does not exists. Creating " + file_complete_path
            if seed_selection_type == 'IS':
                with tqdm(total=len(topic_qrels)) as pbar:
                    for topicId in sorted(topic_qrels.iterkeys()):
                        original_qrel = topic_original_qrels_in_doc_index[topicId]
                        train_index_list = []
                        seed_one_counter = 0.0
                        seed_zero_counter = 0.0
                        for document_index in sorted(original_qrel.iterkeys()):
                            if seed_zero_counter == number_of_seeds/2.0 and seed_one_counter == number_of_seeds/2.0:
                                break
                            doc_label = original_qrel[document_index]
                            if doc_label == 1 and seed_one_counter<(number_of_seeds/2.0):
                                train_index_list.append(document_index)
                                seed_one_counter = seed_one_counter + 1.0
                            elif doc_label == 0 and seed_zero_counter<(number_of_seeds/2.0):
                                train_index_list.append(document_index)
                                seed_zero_counter = seed_zero_counter + 1.0
                        topic_seed_info[topicId] = train_index_list
                        pbar.update()
                pickle.dump(topic_seed_info, open(file_complete_path, 'wb'))
            elif seed_selection_type == 'RDS':
                with tqdm(total=len(topic_qrels)) as pbar:
                    for topicId in sorted(topic_qrels.iterkeys()):
                        original_qrel = topic_original_qrels_in_doc_index[topicId]
                        documentIdList = documentIdListFromRanker[topicId]

                        # there are two special cases we need to handle here
                        # suppose the document collection size is 100, from 0 to 100
                        # but the original_qrels collection contains only 0 to 20 documents
                        # suppose in original qrel we have two topics
                        # topics 1 --> 3,5
                        # topics 2 --> 10,15

                        # Case 1:
                        # For a topic 1, Ranker returns a document outside original qrels like retunr docID 30
                        # so we cannot find that document insides our docIdToDocIndex variable because
                        # that contains only upto documents 20 index

                        # Case 2:
                        # For a topic 1 Ranker returns a document having index 15 that is inside the range of original qrels
                        # and obviously we can find it inside docIdToDocIndex as that contains only upto documents 20 index
                        # and because it is also contains in the original qrels of topic 2 (10,15)
                        # but notice the qrels for that topic 1: contains only topics 1 --> 3,5 so we cannot find that
                        # indside original_qrel[topic 1] because it contains only 3, 5

                        # finally we are loading the seed documents here
                        # so we need those from qrels ONLY

                        documentIndexList = []
                        for document_id in documentIdList:
                            # Handling Case 1 here
                            if document_id in docIdToDocIndex:
                                document_index = docIdToDocIndex[document_id]
                                # Handling Case 2 here
                                # note that original_qrel is the dictionary of document_index
                                if document_index in original_qrel:
                                    documentIndexList.append(document_index)

                        train_index_list = []
                        seed_one_counter = 0.0
                        seed_zero_counter = 0.0
                        for document_index in documentIndexList:
                            if seed_zero_counter >= number_of_seeds / 2.0 and seed_one_counter >= number_of_seeds / 2.0:
                                break
                            doc_label = original_qrel[document_index]
                            if doc_label == 1:
                                train_index_list.append(document_index)
                                seed_one_counter = seed_one_counter + 1.0
                            elif doc_label == 0:
                                train_index_list.append(document_index)
                                seed_zero_counter = seed_zero_counter + 1.0
                        topic_seed_info[topicId] = train_index_list
                        pbar.update()
                pickle.dump(topic_seed_info, open(file_complete_path, 'wb'))

        return topic_seed_info




    # topicId must be string, useTopicDescription is a flag by default it is False
    def get_topic_processed_file(self, topicId, useTopicDescription=False, docrepresentation="TF-IDF"):

        docList = self.topic_to_doclist[str(topicId)]
        relevance_label = []
        judged_review = []

        if useTopicDescription == True:

            # adding the topic description as a relevant document
            judged_review.append(self.topicDescription[str(topicId)])
            relevance_label.append(1)

            # need to add another description of another topic as non-relevant

            nonRelevantTopicId = int(topicId) + 1
            if nonRelevantTopicId > int(self.endTopic):
                nonRelevantTopicId = int(topicId) - 1
            judged_review.append(self.topicDescription[str(nonRelevantTopicId)])
            relevance_label.append(0)

        for docId in docList:
            if docId in self.processedPooledDocuments:
                judged_review.append(self.processedPooledDocuments[docId])
                relevance_label.append(self.docId_label[docId])

        if docrepresentation == "TF-IDF":
            #print "Using TF-IDF"
            vectorizer = TfidfVectorizer(analyzer="word", \
                                         tokenizer=None, \
                                         preprocessor=None, \
                                         stop_words=None, \
                                         max_features=15000)

            bag_of_word = vectorizer.fit_transform(judged_review)


        elif docrepresentation == "BOW":
            #print "Uisng Bag of Word"
            vectorizer = CountVectorizer(analyzer="word", \
                                         tokenizer=None, \
                                         preprocessor=None, \
                                         stop_words=None, \
                                         max_features=15000)

            bag_of_word = vectorizer.fit_transform(judged_review)

        bag_of_word = bag_of_word.toarray()
        relevance_labels = np.asarray(relevance_label)
        return bag_of_word, relevance_labels


'''
t = TRECTopics('TREC8', 401, 450)
t.set_topic_description_path("/work/04549/mustaf/maverick/data/TREC/TREC8/topics.401-450")
t.set_pooled_processed_document_path("/work/04549/mustaf/maverick/data/TREC/TREC8/processed.txt")
t.load_prcessed_document()
t.load_relevance_judgements("/work/04549/mustaf/maverick/data/TREC/TREC8/relevance.txt")
t.process_topic_description()

docContents,docLabels = t.get_topic_processed_file(450,True)

for docContent, docLabel in zip(docContents,docLabels):
    print docContent, docLabel
'''

