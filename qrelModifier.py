#topicSkipList = [202,209,225,237, 245, 255,269, 278, 803, 805] # skip list for others

topicSkipList = [202,225,255,278,805] # For IS seed set # skip_list for JASIST
# jaist skipping WT2013 == 13765, 13717 WT2014 13799 but prediction contains 13735
datasource = 'TREC7' # can be  dataset = ['TREC8', 'gov2', 'WT']
if datasource=='TREC8':
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/relevance.txt'
    start_topic = 401
    end_topic = 451
elif datasource=='TREC7':
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/TREC7/relevance.txt'
    destinationAddress = '/work/04549/mustaf/maverick/data/TREC/gov2/modified_qreldocsgov2_jasist.txt'
    start_topic = 801
    end_topic = 851

elif datasource=='gov2':
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/gov2/qrels.tb06.top50.txt'
    destinationAddress = '/work/04549/mustaf/maverick/data/TREC/gov2/modified_qreldocsgov2_jasist.txt'
    start_topic = 801
    end_topic = 851
elif datasource=='WT2013':
    prediction_file = '/work/04549/mustaf/maverick/data/TREC/deterministic1/WT2013/no_ranker/oversample/prediction_protocol:SAL_batch:25_seed:10_fold1_1.1_modified.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/WT2013/qrelsadhoc2013.txt'
    destinationAddress = '/work/04549/mustaf/maverick/data/TREC/WT2013/modified_qreldocs2013_jasist_new.txt'
    start_topic = 201
    end_topic = 251
else:
    prediction_file = '/work/04549/mustaf/maverick/data/TREC/deterministic1/WT2014/no_ranker/oversample/prediction_protocol:SAL_batch:25_seed:10_fold1_1.1_modified.txt'
    RELEVANCE_DATA_DIR = '/work/04549/mustaf/maverick/data/TREC/WT2014/qrelsadhoc2014.txt'
    destinationAddress = '/work/04549/mustaf/maverick/data/TREC/WT2014/modified_qreldocs2014_jasist_new.txt'
    start_topic = 251
    end_topic = 301


f = open(RELEVANCE_DATA_DIR)
print f
originalqrel = []
one_count = 0
zero_count = 0
line_counter = 0
for lines in f:
    values = lines.split(" ")
    label = int(values[3])
    if label == 1:
        one_count = one_count + 1
    elif label == 0:
        zero_count = zero_count + 1

    line_counter = line_counter + 1
f.close()

print "1:", one_count
print "0:", zero_count
print "total:", line_counter, one_count + zero_count
print "prevalence:", ((one_count*1.0)/(one_count + zero_count))*100

exit()


topic_info = {}
f = open(prediction_file)
line_counter = 0
print f
for lines in f:
    values = lines.split()
    topicNo = values[0]
    docNo = values[2]
    if topicNo not in topic_info:
        doc_list = []
        doc_list.append(docNo)
        topic_info[topicNo] = doc_list
    else:
        doc_list = topic_info[topicNo]
        doc_list.append(docNo)
        topic_info[topicNo] = doc_list
    line_counter = line_counter + 1

f.close()
print "prediction file contains:", line_counter

lineCounter = 0
f = open(RELEVANCE_DATA_DIR)
print f
s = ""
for lines in f:
    values = lines.split()
    topicNo = values[0]
    if topicNo not in topic_info:
        continue
    docNo = values[2]
    doc_list = topic_info[topicNo]
    if docNo not in doc_list:
        continue
    column2 = values[1]
    label = int(values[3])
    if label > 1:
        label = 1
    if label < 0:
        label = 0
    s = s + str(topicNo) + " " + str(column2) + " " + str(docNo) + " " + str(label) + "\n"
    lineCounter = lineCounter + 1
f.close()

print "Modifed Qrel File Contains Line:", lineCounter

'''
lineCounter = 0
modified = 0
f = open(RELEVANCE_DATA_DIR)
print f
originalqrel = []
s=''
for lines in f:
    values = lines.split(" ")
    lineCounter = lineCounter + 1
    topicNo = int(values[0])
    #print topicNo
    if topicNo in topicSkipList:
        continue
    docNo = values[2]
    if docNo not in realqrel:
        continue
    column2 = values[1]
    label = int(values[3])
    if label > 1:
        label = 1
    if label < 0:
        label = 0
    s = s+str(topicNo)+" "+str(column2)+" "+str(docNo)+" "+str(label)+"\n"
    modified = modified + 1
f.close()
print "Original Qrel File Contains Line:", lineCounter
print "Modified Qrel File Contains Line:", modified
'''
output = open(destinationAddress, "w")
output.write(s)
output.close()