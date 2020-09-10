import gzip
import os


#source = ["TREC8", "gov2", "WT2013", "WT2014"]
source = ['TREC7']
for dataset in source:
    address = "/work/04549/mustaf/maverick/data/TREC/"+dataset+"/zippedsystemRankings/"
    destinationBase = "/work/04549/mustaf/maverick/data/TREC/"+dataset+"/systemRankings/"

    fileList = os.listdir(address)

    for fileName in fileList:
        fileAddress = address + fileName
        with gzip.open(fileAddress, 'rb') as f:
            file_content = f.read()
            destinationAddress = destinationBase + os.path.splitext(fileName)[0]
            output = open(destinationAddress, "w")
            output.write(file_content)
            output.close()