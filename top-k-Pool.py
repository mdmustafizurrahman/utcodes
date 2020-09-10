import os

topicSkipList = [202,209,225,237, 245, 255,269, 278, 803, 805] # remember to update the relevance file for this collection accordingly to TAU compute
top_k_list = range(10, 110, 10)

# TREC-8 Adhoc Run List
runs = "input.1 input.8manexT3D1N0 input.acsys8alo input.acsys8amn input.AntHoc1 input.apl8c221 input.apl8n input.att99atdc input.att99atde input.cirtrc82 input.CL99SD input.CL99XT input.disco1 input.Dm8Nbn input.Dm8TFbn input.Flab8as input.Flab8atdn input.fub99a input.fub99tf input.GE8ATDN1 input.ibmg99a input.ibmg99b input.ibms99a input.ibms99b input.ic99dafb input.iit99au1 input.iit99ma1 input.INQ603 input.INQ604 input.isa25 input.isa50 input.kdd8ps16 input.kdd8qe01 input.kuadhoc input.mds08a3 input.mds08a4 input.Mer8Adtd1 input.Mer8Adtd2 input.MITSLStd input.MITSLStdn input.nttd8ale input.nttd8alx input.ok8alx input.ok8amxc input.orcl99man input.pir9Aatd input.pir9Attd input.plt8ah1 input.plt8ah2 input.READWARE input.READWARE2 input.ric8dpx input.ric8tpx input.Sab8A1 input.Sab8A2 input.Scai8Adhoc input.surfahi1 input.surfahi2 input.tno8d3 input.tno8d4 input.umd99a1 input.unc8al32 input.unc8al42 input.UniNET8Lg input.UniNET8St input.UT810 input.UT813 input.uwmt8a1 input.uwmt8a2 input.weaver1 input.weaver2"
run_name = []
for run in runs.split(" "):
    run_name.append(run)

base_address = '/work/04549/mustaf/maverick/data/TREC/'

datasource = 'TREC8' # can be  dataset = ['TREC8', 'gov2', 'WT']
ranker_address = base_address + datasource + '/systemRankings/'
qrel_address = base_address + datasource + '/qrels.trec8.adhoc'
topKpool_base = base_address + datasource + '/topKPoolQrels/'

start_topic = -1
end_topic = -1
datasetsize = 0

if datasource == 'TREC8':
    start_topic = 401
    end_topic = 451
    datasetsize = 86830

print('Reading the relevance label')

topic_to_doclist = {}  # key is the topic(string) and value is the dictionary where key is doc, value is label
docIndex_DocNo = {}  # key is the index used in my code value is the actual DocNo
docNo_docIndex = {}  # key is the DocNo and the value is the index assigned by my code

for topic in xrange(start_topic, end_topic):
    f = open(qrel_address)
    tmpDict = {}
    for lines in f:
        values = lines.split()
        topicNo = values[0]
        if int(topicNo) != topic:
            continue
        docNo = values[2]
        label = int(values[3])
        if label >= 1:
            label = 1
        elif label <= 0:
            label = 0
        if (topic_to_doclist.has_key(topicNo)):
            tmpDict = topic_to_doclist[topicNo]
            tmpDict[docNo] = label
            topic_to_doclist[topicNo] = tmpDict
        else:
            tmpDict = {}
            tmpDict[docNo] = label
            topic_to_doclist[topicNo] = tmpDict
    f.close()

number_of_docs = 0
for topic, topicDocList in topic_to_doclist.iteritems():
    #print topic, len(topicDocList)
    number_of_docs = number_of_docs + len(topicDocList)

#print "Total Docs in " + datasource +" = " + str(number_of_docs)

# reading rankers
fileList = sorted(os.listdir(ranker_address))
rank_info = {} # key is the 'ranker_name', value is a dictionary ('topicid', dictionary(rank, 'docNo'))
rankerNameCount = 0
for rankerName in fileList:

    if rankerName not in run_name:
        continue

    fileAddress = ranker_address + rankerName
    f = open(fileAddress)
    # print fileAddress

    topicInfo = {} # key--TopicID values -- Dictionary(rank, )
    for lines in f:
        values = lines.split("\t")
        topicNo = values[0]
        docNo = values[2]
        rank_no = int(values[3])

        if topicNo in topicInfo:
            rank_to_docNo = topicInfo[topicNo]
            rank_to_docNo[rank_no] = docNo
            topicInfo[topicNo] = rank_to_docNo
        else:
            rank_to_docNo = {}
            rank_to_docNo[rank_no] = docNo
            topicInfo[topicNo] = rank_to_docNo

    rank_info[rankerName] = topicInfo
    rankerNameCount+= 1

print "Number of Ranks:", rankerNameCount

#exit(0)

s=""

for k in top_k_list:
    top_k_pool_tmp = []


    for topicIndex in xrange(start_topic, end_topic):
        topicNo = str(topicIndex)
        top_k_pool_doc_list = []
        for rank_name, topicInfo in rank_info.iteritems():
            tmpranktodoc =  topicInfo[topicNo]
            i = 0
            for i, key in enumerate(sorted(tmpranktodoc.iterkeys())):
                docNo = tmpranktodoc[key]
                if docNo not in top_k_pool_doc_list:
                    if docNo in topic_to_doclist[topicNo]:
                        label = topic_to_doclist[topicNo][docNo]
                        top_k_pool_tmp.append((topicNo,docNo, label))
                        top_k_pool_doc_list.append(docNo)
                    #else:
                        # print rank_name, topicNo, key, docNo
                        #label = 0 # not judged so non relevant from TREC-8 pool paper
                        #top_k_pool_tmp.append((topicNo, docNo, label))
                        #top_k_pool_doc_list.append(docNo)

                if i> (k - 1):
                    break
    print "top:", str(k), len(top_k_pool_tmp)

    s=""
    for item in top_k_pool_tmp:
        s=s+item[0]+" 0 "+item[1]+" "+str(item[2])+"\n"

    text_file = open(topKpool_base+"pool_"+str(k)+".txt", "w")
    text_file.write(s)
    text_file.close()


