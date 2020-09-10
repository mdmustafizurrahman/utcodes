import os

topicSkipList = [202,209,225,237, 245, 255,269, 278, 803, 805] # remember to update the relevance file for this collection accordingly to TAU compute

datasource = 'WT2013' # can be  dataset = ['TREC8', 'gov2', 'WT']
if datasource=='WT2013':
    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/"+datasource+"/"
    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modified_qreldocs2013.txt'
    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"

else:
    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modified_qreldocs2014.txt'
    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"


lineCounter = 0
f = open(qrelAdress)
print f
modifiedqrel = []
for lines in f:
    values = lines.split()
    docNo = values[2]
    #print docNo
    modifiedqrel.append(docNo)
    lineCounter = lineCounter + 1
f.close()



fileList = os.listdir(originAdress)
for fileName in fileList:
    fileAddress = originAdress + fileName
    f = open(fileAddress)
    print fileAddress
    s=''
    lineCounter = 0
    modified = 0
    miss = 0
    for lines in f:
        values = lines.split("\t")
        lineCounter = lineCounter + 1
        topicNo = int(values[0])

        if topicNo == 202:
            continue
        docNo = values[2]
        #print docNo
        if docNo not in modifiedqrel:
            miss = miss + 1
            print "Missing", docNo
            continue
        '''column2 = values[1]
        label = int(values[3])
        column4 = values[4]
        column5 = values[5]
        if label > 1:
            label = 1
        if label < 0:
            label = 0
        '''
        #s = s+str(topicNo)+"\t"+str(column2)+"\t"+str(docNo)+"\t"+str(label)+"\t"+str(column4)+"\t"+str(column5)
        s = s + lines
        modified = modified + 1

    f.close()
    destinationAddress = destinationBase + fileName
    output = open(destinationAddress, "w")
    output.write(s)
    output.close()
    print lineCounter, modified, miss
    exit()

