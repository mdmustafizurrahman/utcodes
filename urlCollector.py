from bs4 import BeautifulSoup

import urllib2, base64
import os
import requests
from requests.auth import HTTPBasicAuth

dataset = "TREC7"

if dataset == "TREC-8":
    SystemRankingAddress = "http://trec.nist.gov/results/trec8/trec8.results.input/index.html"
    RankingBaseAddress = "http://trec.nist.gov/results/trec8/trec8.results.input/"
elif dataset == "TREC7":
    SystemRankingAddress = "https://trec.nist.gov/results/trec7/trec7.results.input/index.html"
    RankingBaseAddress = "https://trec.nist.gov/results/trec7/trec7.results.input/"

elif dataset == "gov2":
    SystemRankingAddress = "http://trec.nist.gov/results/trec15/terabyte-adhoc.input.html"
    RankingBaseAddress = "http://trec.nist.gov/results/trec15/"
elif dataset == "WT2013":
    SystemRankingAddress = "http://trec.nist.gov/results/trec22/web.input.html"
    RankingBaseAddress = "http://trec.nist.gov/results/trec22/"
elif dataset == "WT2014":
    SystemRankingAddress = "http://trec.nist.gov/results/trec23/web.adhoc.input.html"
    RankingBaseAddress = "http://trec.nist.gov/results/trec23/"

request = urllib2.Request(SystemRankingAddress)
base64string = base64.encodestring('%s:%s' % ('tipster', 'cdroms')).replace('\n', '')
request.add_header("Authorization", "Basic %s" % base64string)
resp = urllib2.urlopen(request)
destinationAddress = "/work/04549/mustaf/maverick/data/TREC/"+dataset+"/zippedsystemRankings/"

soup = BeautifulSoup(resp, from_encoding=resp.info().getparam('charset'))

for link in soup.find_all('a', href=True):
    address = str(link['href'])
    #print address
    if dataset == "TREC-8" or dataset == "TREC7":
        if dataset == "TREC8" and address.find("adhoc")>=0:
            downloadAddress = RankingBaseAddress+address
            fileName =  os.path.basename(downloadAddress)
            print fileName
            response = urllib2.urlopen(request)
            # use verify=False to ignore the SSLError
            r = requests.get(downloadAddress, auth=HTTPBasicAuth('tipster', 'cdroms'), verify=False)
            output = open(destinationAddress+fileName, "w")
            output.write(r.content)
            output.close()
        if dataset == 'TREC7':
            if address.find("adhoc")>=0 or address.find("high_prec")>=0:
                downloadAddress = RankingBaseAddress + address
                fileName = os.path.basename(downloadAddress)
                print fileName
                response = urllib2.urlopen(request)
                # use verify=False to ignore the SSLError
                r = requests.get(downloadAddress, auth=HTTPBasicAuth('tipster', 'cdroms'), verify=False)
                output = open(destinationAddress + fileName, "w")
                output.write(r.content)
                output.close()



    elif dataset == "gov2":
        if address.find("terabyte") >= 0:
            print address
            downloadAddress = RankingBaseAddress + address
            fileName = os.path.basename(downloadAddress)

            response = urllib2.urlopen(request)
            r = requests.get(downloadAddress, auth=HTTPBasicAuth('tipster', 'cdroms'))
            output = open(destinationAddress + fileName, "w")
            output.write(r.content)
            output.close()
    elif dataset == "WT2013" or dataset == "WT2014":
        if address.find("web") >= 0:
            print address
            downloadAddress = RankingBaseAddress + address
            fileName = os.path.basename(downloadAddress)
            response = urllib2.urlopen(request)
            r = requests.get(downloadAddress, auth=HTTPBasicAuth('tipster', 'cdroms'))
            output = open(destinationAddress + fileName, "w")
            output.write(r.content)
            output.close()
