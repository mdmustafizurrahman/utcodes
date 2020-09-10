import numpy as np
import sys
import copy
from math import log
import math
from numpy import trapz

base_address = "/work/04549/mustaf/lonestar/data/TREC/"
trec_eval_executable = base_address + 'trec_eval.9.0/trec_eval'

data_set_name_list = ['Adhoc\'98','Adhoc\'99', 'TB\'06', 'WT\'13', 'WT\'14']
qrelAddress = {}
qrelAddress['TREC8'] = base_address + 'TREC8/relevance.txt'
qrelAddress['TREC7'] = base_address + 'TREC7/relevance.txt'
qrelAddress['gov2'] = base_address + 'gov2/qrels.tb06.top50.txt'
#qrelAddress['WT2013'] = base_address + 'WT2013/modified_qreldocs2013.txt'
#qrelAddress['WT2014'] = base_address + 'WT2014/modified_qreldocs2014.txt'
qrelAddress['WT2013'] = base_address + 'WT2013/qrelsadhoc2013.txt'
qrelAddress['WT2014'] = base_address + 'WT2014/qrelsadhoc2014.txt'


datasouce_name_toacronym = {}
datasouce_name_toacronym['TREC8'] = 'Adhoc\'99'
datasouce_name_toacronym['TREC7'] = 'Adhoc\'98'
datasouce_name_toacronym['gov2'] = 'TB\'06'
datasouce_name_toacronym['WT2013'] = 'WT\'13'
datasouce_name_toacronym['WT2014'] = 'WT\'14'




#rankMetric: map, infAP, P.10

systemAddress = {}
systemAddress['TREC8'] = base_address + 'TREC8/systemRankings/'
systemAddress['TREC7'] = base_address + 'TREC7/systemRankings/'
systemAddress['gov2'] = base_address + 'gov2/systemRankings/'
systemAddress['WT2013'] = base_address + 'WT2013/systemRankings/'
systemAddress['WT2014'] = base_address + 'WT2014/systemRankings/'

systemName = {}
systemName['TREC8'] = 'input.ibmg99b'
systemName['TREC7'] = 'input.dsir07a02'
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
#start_topic['TREC8'] = 450
start_topic['TREC7'] = 351
#start_topic['TREC7'] = 373
start_topic['gov2'] = 801
start_topic['WT2013'] = 201
start_topic['WT2014'] = 251

end_topic = {}
end_topic['TREC8'] = 451
end_topic['TREC7'] = 401
end_topic['gov2'] = 851
end_topic['WT2013'] = 251
end_topic['WT2014'] = 301

# for TREC8 and TREC 7 2-20 ; 21-29; 30-36; 37-40
group_considered_start = {}
group_considered_start['TREC8'] = 2
group_considered_start['TREC7'] = 2
group_considered_start['gov2'] = 2
group_considered_start['WT2013'] = 2
group_considered_start['WT2014'] = 2


group_considered_step = {}
group_considered_step['TREC8'] = 1
group_considered_step['TREC7'] = 1
group_considered_step['gov2'] = 1
group_considered_step['WT2013'] = 1
group_considered_step['WT2014'] = 1

def update_run_name(list1):
    tmp = []
    for run in list1:
        if '-' in run: # WT2013 runs
            run = run.replace('-', '_')
        tmp.append('input.'+run)
    return tmp

'''
group_considered_start = {}
group_considered_start['TREC8'] = 10
group_considered_start['TREC7'] = 10
group_considered_start['gov2'] = 5
group_considered_start['WT2013'] = 4
group_considered_start['WT2014'] = 3


group_considered_step = {}
group_considered_step['TREC8'] = 10
group_considered_step['TREC7'] = 10
group_considered_step['gov2'] = 5
group_considered_step['WT2013'] = 4
group_considered_step['WT2014'] = 3

'''

qrelSize = {}
qrelSize['TREC8'] = 86830
qrelSize['TREC7'] = 80345
topicSkipList = [202,209,225,237, 245, 255,269, 278, 803, 805] # remember to update the relevance file for this collection accordingly to TAU compute

collection_size = {}
collection_size['TREC8'] = 528155
collection_size['TREC7'] = 528155

prevalence_ratio = {}
prevalence_ratio['TREC8'] = (qrelSize['TREC8']*1.0)/collection_size['TREC8']
prevalence_ratio['TREC7'] = (qrelSize['TREC7']*1.0)/collection_size['TREC7']

# total 71 runs in pool
# https://tec.citius.usc.es/ir/code/pooling_bandits.html
system_runs_TREC8 = "input.1 input.8manexT3D1N0 input.acsys8alo " \
                    "input.acsys8amn input.AntHoc1 input.apl8c221 input.apl8n " \
                    "input.att99atdc input.att99atde input.cirtrc82 input.CL99SD " \
                    "input.CL99XT input.disco1 input.Dm8Nbn input.Dm8TFbn input.Flab8as " \
                    "input.Flab8atdn input.fub99a input.fub99tf input.GE8ATDN1 input.ibmg99a " \
                    "input.ibmg99b input.ibms99a input.ibms99b input.ic99dafb input.iit99au1 " \
                    "input.iit99ma1 input.INQ603 input.INQ604 input.isa25 input.isa50 " \
                    "input.kdd8ps16 input.kdd8qe01 input.kuadhoc input.mds08a3 input.mds08a4 " \
                    "input.Mer8Adtd1 input.Mer8Adtd2 input.MITSLStd input.MITSLStdn " \
                    "input.nttd8ale input.nttd8alx input.ok8alx input.ok8amxc input.orcl99man " \
                    "input.pir9Aatd input.pir9Attd input.plt8ah1 input.plt8ah2 input.READWARE " \
                    "input.READWARE2 input.ric8dpx input.ric8tpx input.Sab8A1 input.Sab8A2 " \
                    "input.Scai8Adhoc input.surfahi1 input.surfahi2 input.tno8d3 input.tno8d4 " \
                    "input.umd99a1 input.unc8al32 input.unc8al42 input.UniNET8Lg input.UniNET8St " \
                    "input.UT810 input.UT813 input.uwmt8a1 input.uwmt8a2 input.weaver1 " \
                    "input.weaver2"



group_list_TREC8 = {}
group_list_TREC8[1] = ['input.1']
group_list_TREC8[2] = ['input.8manexT3D1N0']
group_list_TREC8[3] = ['input.acsys8alo', 'input.acsys8amn']
group_list_TREC8[4] = ['input.AntHoc1']
group_list_TREC8[5] = ['input.apl8c221', 'input.apl8n']
group_list_TREC8[6] = ['input.att99atdc', 'input.att99atde']
group_list_TREC8[7] = ['input.cirtrc82']
group_list_TREC8[8] = ['input.CL99SD', 'input.CL99XT']
group_list_TREC8[9] = ['input.disco1']
group_list_TREC8[10] = ['input.Dm8Nbn', 'input.Dm8TFbn']
group_list_TREC8[11] = ['input.Flab8as', 'input.Flab8atdn']
group_list_TREC8[12] = ['input.fub99a', 'input.fub99tf']
group_list_TREC8[13] = ['input.GE8ATDN1']
group_list_TREC8[14] = ['input.ibmg99a', 'input.ibmg99b', 'input.ibms99a', 'input.ibms99b']
group_list_TREC8[15] = ['input.ic99dafb']
group_list_TREC8[16] = ['input.iit99au1', 'input.iit99ma1']
group_list_TREC8[17] = ['input.INQ603', 'input.INQ604']
group_list_TREC8[18] = ['input.isa25', 'input.isa50']
group_list_TREC8[19] = ['input.kdd8ps16', 'input.kdd8qe01']
group_list_TREC8[20] = ['input.kuadhoc']
group_list_TREC8[21] = ['input.mds08a3', 'input.mds08a4']
group_list_TREC8[22] = ['input.Mer8Adtd1', 'input.Mer8Adtd2']
group_list_TREC8[23] = ['input.MITSLStd', 'input.MITSLStdn']
group_list_TREC8[24] = ['input.nttd8ale', 'input.nttd8alx']
group_list_TREC8[25] = ['input.ok8alx', 'input.ok8amxc']
group_list_TREC8[26] = ['input.orcl99man']
group_list_TREC8[27] = ['input.pir9Aatd', 'input.pir9Attd']
group_list_TREC8[28] = ['input.plt8ah1', 'input.plt8ah2']
group_list_TREC8[29] = ['input.READWARE', 'input.READWARE2']
group_list_TREC8[30] = ['input.ric8dpx', 'input.ric8tpx']
group_list_TREC8[31] = ['input.Sab8A1', 'input.Sab8A2']
group_list_TREC8[32] = ['input.Scai8Adhoc']
group_list_TREC8[33] = ['input.surfahi1', 'input.surfahi2']
group_list_TREC8[34] = ['input.tno8d3', 'input.tno8d4']
group_list_TREC8[35] = ['input.umd99a1']
group_list_TREC8[36] = ['input.unc8al32', 'input.unc8al42']
group_list_TREC8[37] = ['input.UniNET8Lg', 'input.UniNET8St']
group_list_TREC8[38] = ['input.UT810', 'input.UT813']
group_list_TREC8[39] = ['input.uwmt8a1', 'input.uwmt8a2']
group_list_TREC8[40] = ['input.weaver1', 'input.weaver2']



rnus_not_in_WT_2014_adhoc = ['UDInfoWebRiskTR','UDInfoWebRiskRM','UDInfoWebRiskAX',
                             'ICTNET14RSR1','ICTNET14RSR2','uogTrq1','uogTrBwf','ICTNET14RSR3',
                             'udelCombCAT2','uogTrDwsts','wistud.runD','wistud.runE']

system_runs_TREC8_list = []
for system_name in sorted(system_runs_TREC8.split(" ")):
    system_runs_TREC8_list.append(system_name)

# 84 runs in the pool (77 adhoc + 7 other).
system_runs_TREC7 = "input.acsys7al input.acsys7mi input.AntHoc01 input.APL985LC input.APL985SC " \
                    "input.att98atdc input.att98atde input.bbn1 input.Brkly25 input.Brkly26 " \
                    "input.CLARIT98CLUS input.CLARIT98COMB input.Cor7A1clt input.Cor7A3rrf " \
                    "input.dsir07a01 input.dsir07a02 input.ETHAC0 input.ETHAR0 input.FLab7ad " \
                    "input.FLab7at input.fsclt7a input.fsclt7m input.fub98a input.fub98b " \
                    "input.gersh1 input.gersh2 input.harris1 input.ibmg98a input.ibmg98b " \
                    "input.ibms98a input.ibms98b input.ic98san3 input.ic98san4 input.iit98au1 " \
                    "input.iit98ma1 input.INQ501 input.INQ502 input.iowacuhk1 input.iowacuhk2 " \
                    "input.jalbse011 input.jalbse012 input.KD70000 input.KD71010s input.kslsV1 " \
                    "input.lanl981 input.LIArel2 input.LIAshort2 input.LNaTitDesc7 input.LNmanual7 " \
                    "input.mds98t input.mds98td input.MerAdRbtnd input.MerTetAdtnd input.nectitech " \
                    "input.nectitechdes input.nsasgrp3 input.nsasgrp4 input.nthu1 input.nthu2 " \
                    "input.nttdata7Al0 input.nttdata7Al2 input.ok7am input.ok7ax input.pirc8Aa2 " \
                    "input.pirc8Ad input.ScaiTrec7 input.t7miti1 input.tno7exp1 input.tno7tw4 " \
                    "input.umd98a1 input.umd98a2 input.unc7aal1 input.unc7aal2 input.uoftimgr " \
                    "input.uoftimgu input.uwmt7a1 input.uwmt7a2 input.acsys7hp " \
                    "input.Cor7HP1 input.Cor7HP2 input.Cor7HP3 input.pirc8Ha input.uwmt7h1 input.uwmt7h2"


group_list_TREC7 = {}
group_list_TREC7[1] = ["input.acsys7al", "input.acsys7mi", "input.acsys7hp"]
group_list_TREC7[2] = ["input.AntHoc01"]
group_list_TREC7[3] = ["input.APL985LC", "input.APL985SC"]
group_list_TREC7[4] = ["input.att98atdc", "input.att98atde"]
group_list_TREC7[5] = ["input.bbn1"]
group_list_TREC7[6] = ["input.Brkly25", "input.Brkly26"]
group_list_TREC7[7] = ["input.CLARIT98CLUS", "input.CLARIT98COMB"]
group_list_TREC7[8] = ["input.Cor7A1clt", "input.Cor7A3rrf", "input.Cor7HP1", "input.Cor7HP2", "input.Cor7HP3"]
group_list_TREC7[9] = ["input.dsir07a01", "input.dsir07a02"]
group_list_TREC7[10] = ["input.ETHAC0", "input.ETHAR0"]
group_list_TREC7[11] = ["input.FLab7ad", "input.FLab7at"]
group_list_TREC7[12] = ["input.fsclt7a", "input.fsclt7m"]
group_list_TREC7[13] = ["input.fub98a", "input.fub98b"]
group_list_TREC7[14] = ["input.gersh1", "input.gersh2"]
group_list_TREC7[15] = ["input.harris1"]
group_list_TREC7[16] = ["input.ibmg98a", "input.ibmg98b", "input.ibms98a", "input.ibms98b"]
group_list_TREC7[17] = ["input.ic98san3", "input.ic98san4"]
group_list_TREC7[18] = ["input.iit98au1", "input.iit98ma1"]
group_list_TREC7[19] = ["input.INQ501", "input.INQ502"]
group_list_TREC7[20] = ["input.iowacuhk1", "input.iowacuhk2"]
group_list_TREC7[21] = ["input.jalbse011", "input.jalbse012"]
group_list_TREC7[22] = ["input.KD70000", "input.KD71010s"]
group_list_TREC7[23] = ["input.kslsV1"]
group_list_TREC7[24] = ["input.lanl981"]
group_list_TREC7[25] = ["input.LIArel2", "input.LIAshort2"]
group_list_TREC7[26] = ["input.LNaTitDesc7", "input.LNmanual7"]
group_list_TREC7[27] = ["input.mds98t", "input.mds98td"]
group_list_TREC7[28] = ["input.MerAdRbtnd", "input.MerTetAdtnd"]
group_list_TREC7[29] = ["input.nectitech", "input.nectitechdes"]
group_list_TREC7[30] = ["input.nsasgrp3", "input.nsasgrp4"]
group_list_TREC7[31] = ["input.nthu1", "input.nthu2"]
group_list_TREC7[32] = ["input.nttdata7Al0", "input.nttdata7Al2"]
group_list_TREC7[33] = ["input.ok7am", "input.ok7ax"]
group_list_TREC7[34] = ["input.pirc8Aa2", "input.pirc8Ad", "input.pirc8Ha"]
group_list_TREC7[35] = ["input.ScaiTrec7"]
group_list_TREC7[36] = ["input.t7miti1"]
group_list_TREC7[37] = ["input.tno7exp1", "input.tno7tw4"]
group_list_TREC7[38] = ["input.umd98a1", "input.umd98a2"]
group_list_TREC7[39] = ["input.unc7aal1", "input.unc7aal2"]
group_list_TREC7[40] = ["input.uoftimgr", "input.uoftimgu"]
group_list_TREC7[41] = ["input.uwmt7a1", "input.uwmt7a2", "input.uwmt7h1", "input.uwmt7h2"]


group_list_TB = {}
group_list_TB[1] = {"input.AMRIMtp20006", "input.AMRIMtp5006", "input.AMRIMtpm5006"}
group_list_TB[2] = {"input.arscDomAlog" ,"input.arscDomAsrt", "input.arscDomManL", "input.arscDomManS"}
group_list_TB[3] = {"input.CoveoRun1"}
group_list_TB[4] = {"input.CWI06DISK1ah", "input.CWI06DIST8ah"}
group_list_TB[5] = {"input.DCU05BASE"}
group_list_TB[6] = {"input.hedge0", "input.hedge10", "input.hedge30", "input.hedge50", "input.hedge5"}
group_list_TB[7] = {"input.humT06l", "input.humT06xlc", "input.humT06xle", "input.humT06xl", "input.humT06xlz"}
group_list_TB[8] = {"input.indri06AdmD", "input.indri06AlceB", "input.indri06AlceD", "input.indri06Aql", "input.indri06AtdnD"}
group_list_TB[9] = {"input.JuruMan", "input.JuruTD", "input.JuruT", "input.JuruTWE"}
group_list_TB[10] = {"input.mg4jAdhocBBV", "input.mg4jAdhocBV", "input.mg4jAdhocBVV", "input.mg4jAdhocV",
                     "input.mg4jAutoBBV", "input.mg4jAutoBV", "input.mg4jAutoBVV", "input.mg4jAutoV"}
group_list_TB[11] = {"input.mpiircomb", "input.mpiirdesc", "input.mpiirmanual", "input.mpiirtitle"}
group_list_TB[12] = {"input.MU06TBa1", "input.MU06TBa2", "input.MU06TBa5", "input.MU06TBa6"}
group_list_TB[13] = {"input.p6tbadt", "input.p6tbaxl"}
group_list_TB[14] = {"input.sabtb06aa1", "input.sabtb06at1", "input.sabtb06man1"}
group_list_TB[15] = {"input.THUADALL", "input.THUADAO", "input.THUADLMAO", "input.THUADLMO", "input.THUADOR"}
group_list_TB[16] = {"input.TWTB06AD01", "input.TWTB06AD02", "input.TWTB06AD03", "input.TWTB06AD04", "input.TWTB06AD05"}
group_list_TB[17] = {"input.UAmsT06a3SUM", "input.UAmsT06aAnLM", "input.UAmsT06aTDN", "input.UAmsT06aTeLM", "input.UAmsT06aTTDN"}
group_list_TB[18] = {"input.uogTB06QET1", "input.uogTB06QET2", "input.uogTB06S50L", "input.uogTB06SS10L", "input.uogTB06SSQL"}
group_list_TB[19] = {"input.uwmtFadDS", "input.uwmtFadTPFB", "input.uwmtFadTPRR", "input.uwmtFmanual"}
group_list_TB[20] = {"input.zetabm", "input.zetadir", "input.zetaman", "input.zetamerg2", "input.zetamerg"}


#
#https://trec.nist.gov/pubs/trec22/appendices/web.html
#only web adhoc run list https://trec.nist.gov/pubs/trec22/appendices/web_appendix.pdf
group_list_WT2013 = {}
group_list_WT2013[1] = {"input.ICTNET13ADR1", "input.ICTNET13ADR2", "input.ICTNET13ADR3"}
group_list_WT2013[2] = {"input.UDInfolabWEB1", "input.UDInfolabWEB2"}
group_list_WT2013[3] = {"input.UJS13LCRAd1", "input.UJS13LCRAd2"}
group_list_WT2013[4] = {"input.mmrbf", "input.clustmrfaf", "input.clustmrfbf"}
group_list_WT2013[5] = {"input.cwiwt13cpe", "input.cwiwt13cps", "input.cwiwt13kld"}
group_list_WT2013[6] = {"input.dlde"}
group_list_WT2013[7] = {"input.msr_alpha0_95_4", "input.msr_alpha0"}
group_list_WT2013[8] = {"input.udelCombUD",  "input.udelPseudo1", "input.udelPseudo2"}
group_list_WT2013[9] = {"input.udemQlm1l", "input.udemQlm1lFb", "input.udemQlm1lFbWiki"}
group_list_WT2013[10] = {"input.uogTrAIwLmb", "input.uogTrAS2Lb", "input.uogTrBDnLmxw"}
group_list_WT2013[11] = {"input.ut22base", "input.ut22spam", "input.ut22xact"}
group_list_WT2013[12] = {"input.webismixed", "input.webisrandom", "input.webiswtbaseline"}
group_list_WT2013[13] = {"input.wistud.runA", "input.wistud.runB", "input.wistud.runC"}

risk_run = {}

risk_run_WT2013 = {}
risk_run_WT2013[1] = {"input.ICTNET13RSR1", "input.ICTNET13RSR2", "input.ICTNET13RSR3"}
risk_run_WT2013[2] = {"input.msr_alpha1", "input.msr_alpha10", "input.msr_alpha5"}
risk_run_WT2013[3] = {"input.RMITSC", "input.RMITSC75", "input.RMITSCTh"}
risk_run_WT2013[4] = {"input.udelManExp", "input.udelPseudo1LM"}
risk_run_WT2013[5] = {"input.udemFbWikiR", "input.udemQlml1FbR", "input.udemQlml1R"}
risk_run_WT2013[6] = {"input.UDInfolabWEB1R", "input.UDInfolabWEB2R"}
risk_run_WT2013[7] = {"input.UJS13Risk1", "input.UJS13Risk2"}
risk_run_WT2013[8] = {"input.uogTrADnLrb", "input.uogTrAS1Lb", "input.uogTrBDnLaxw", "input.UWCWEB13RISK01", "input.UWCWEB13RISK02", "input.webishybrid", "input.webisnaive", "input.webiswikibased", "input.wistud.runD"}




#https://trec.nist.gov/pubs/trec23/papers/overview-web.pdf
#total 30 Adhoc runs
#https://trec.nist.gov/pubs/trec23/appendices/web-adhoc_appendix.pdf
group_list_WT2014 = {}
group_list_WT2014[1] = {"input.CiirAll1", "input.CiirSdm", "input.CiirSub1", "input.CiirSub2", "input.CiirWikiRm"}
group_list_WT2014[2] = {"input.ICTNET14ADR1", "input.ICTNET14ADR2", "input.ICTNET14ADR3"}
group_list_WT2014[3] = {"input.Protoss", "input.Terran", "input.Zerg"}
group_list_WT2014[4] = {"input.SNUMedinfo11", "input.SNUMedinfo12", "input.SNUMedinfo13"}
group_list_WT2014[5] = {"input.UDInfoWebAX", "input.UDInfoWebENT", "input.UDInfoWebLES", "input.udel_itu", "input.udel_itub"}
group_list_WT2014[6] = {"input.uogTrDuax", "input.uogTrDwl", "input.uogTrIwa"}
group_list_WT2014[7] = {"input.utbase", "input.utexact"}
group_list_WT2014[8] = {"input.webisWt14axAll", "input.webisWt14axMax", "input.webisWt14axSyn"}
group_list_WT2014[9] = {"input.wistud.runA", "input.wistud.runB", "input.wistud.runC"}


group_list = {}
group_list['TREC8'] = group_list_TREC8
group_list['TREC7'] = group_list_TREC7
group_list['gov2'] = group_list_TB
group_list['WT2013'] = group_list_WT2013
group_list['WT2014'] = group_list_WT2014


def count_number_of_relevant(file_name):

    relevant_count = 0
    f = open(file_name)
    for lines in f:
        #print lines
        values = lines.split(" ")
        label = int(values[3])
        if label == 1:
            #print lines, label
            relevant_count = relevant_count + 1
    return relevant_count



def find_group_number(datasource, run_name):
    datasource_groups = group_list[datasource]
    for group_number, system_name_list in sorted(datasource_groups.iteritems()):
        if run_name in system_name_list:
            return group_number


def generate_manural_run_free_group_list(group_list_for_datasource, datasource):
    new_group_list = {}
    new_grp_no = 1

    for grp_no, run_list in sorted(group_list_for_datasource.iteritems()):
        new_run_list = []
        for run_name in run_list:
            if run_name not in manual_run_list[datasource]:
                new_run_list.append(run_name)
        if len(new_run_list) == 0:
            continue
        new_group_list[new_grp_no] = new_run_list
        new_grp_no = new_grp_no + 1
    return new_group_list



system_runs_TREC7_list = []
for system_name in sorted(system_runs_TREC7.split(" ")):
    system_runs_TREC7_list.append(system_name)


system_runs_TB06_list = []
for group_number, group_members in group_list_TB.iteritems():
    for group_member in group_members:
        system_runs_TB06_list.append(group_member)
system_runs_TB06_list = sorted(system_runs_TB06_list)


system_runs_WT2013_list = []
for group_number, group_members in group_list_WT2013.iteritems():
    for group_member in group_members:
        system_runs_WT2013_list.append(group_member)
system_runs_WT2013_list = sorted(system_runs_WT2013_list)


system_runs_WT2014_list = []
for group_number, group_members in group_list_WT2014.iteritems():
    for group_member in group_members:
        system_runs_WT2014_list.append(group_member)
system_runs_WT2014_list = sorted(system_runs_WT2014_list)



systemNameList = {}
systemNameList['TREC8'] = system_runs_TREC8_list
systemNameList['TREC7'] = system_runs_TREC7_list
systemNameList['gov2'] = system_runs_TB06_list
systemNameList['WT2013'] = system_runs_WT2013_list
systemNameList['WT2014'] = system_runs_WT2014_list


pool_depth = {}
pool_depth['TREC7'] = 100
pool_depth['TREC8'] = 100
pool_depth['gov2'] = 50
pool_depth['WT2013'] = 20
pool_depth['WT2014'] = 25


pool_depth_variation = {}
pool_depth_variation['TREC7'] = [20, 40, 60, 80, 100]
pool_depth_variation['TREC8'] = [20, 40, 60, 80, 100]
pool_depth_variation['gov2'] = [20, 40, 50]
pool_depth_variation['WT2013'] = [20]
pool_depth_variation['WT2014'] = [25]




manual_run_list = {}
'''
# all submitted manual runs
manual_run_list['TREC7'] = ['acsys7mi', 'CLARIT98CLUS', 'CLARIT98COMB', 'CLARIT98RANK', 'fsclt7m', 'gersh1',
                            'iit98ma1', 'harris1', 'LNmanual7', 'lanl981', 't7miti1', 'nthu1', 'Brkly26',
                            'uoftimgr', 'uoftimgu', 'uwmt7a1', 'uwmt7a2']
'''
# participating manual runs in the pool
manual_run_list['TREC7'] = ['acsys7mi', 'CLARIT98CLUS', 'CLARIT98COMB', 'fsclt7m', 'gersh1',
                            'iit98ma1', 'harris1', 'LNmanual7', 'lanl981', 't7miti1', 'nthu1', 'Brkly26',
                            'uoftimgr', 'uoftimgu', 'uwmt7a1', 'uwmt7a2']

manual_run_list['TREC7'] = update_run_name(manual_run_list['TREC7'])

'''
# all submitted manual runs
manual_run_list['TREC8'] = ['input.CL99XTopt', 'input.CL99XT', 'input.CL99SD', 'input.CL99SDopt1', 'input.iit99ma1',
                            'input.orcl99man', 'input.CL99SDopt2', 'input.GE8MTD2', 'input.READWARE2',
                            'input.8manexT3D1N0', 'input.READWARE', 'input.cirtrc82', 'input.disco1']
'''
# participating manual runs in the pool
manual_run_list['TREC8'] = ['input.CL99XT', 'input.CL99SD', 'input.iit99ma1',
                            'input.orcl99man', 'input.READWARE2',
                            'input.8manexT3D1N0', 'input.READWARE', 'input.cirtrc82', 'input.disco1']


manual_run_list['gov2'] = ['AMRIMtpm5006', 'arscDomManL', 'arscDomManS', 'hedge10', 'hedge30', 'hedge50', 'hedge5', 'JuruMan', 'mg4jAdhocBBV', 'mg4jAdhocBV', 'mg4jAdhocBVV','mg4jAdhocV', 'mpiirmanual', 'MU06TBa1','sabtb06man1','TWTB06AD02', 'TWTB06AD03', 'uwmtFmanual', 'zetaman']
manual_run_list['gov2'] = update_run_name(manual_run_list['gov2'])

manual_run_list['WT2013'] = ['dlde', 'msr-alpha0', 'msr-alpha0-95-4']
manual_run_list['WT2013'] = update_run_name(manual_run_list['WT2013'])

manual_run_list['WT2014'] = ['UDInfoWebENT', 'UDInfoWebLES', 'CiirSdm', 'CiirWikiRm']
manual_run_list['WT2014'] = update_run_name(manual_run_list['WT2014'])

budget_manual_list = {}
# from EM Vorohees's paper
budget_manual_list['TREC8'] = [10134, 19231, 28125, 36947, 45584, 54071, 62476, 70654, 78805]

# save_path and data_path is in falcon server
# login into falcon server via fiat
# ssh nahid@fiat.ischool.utexas.edu
# ssh falcon
# data_path = /export/home/u16/nahid/data/vectorizedTREC/TREC8/
# save_path = "/export/home/u16/nahid/data/sqliteDB/"

dictionary_name = "TREC8_dictionary.txt.bz2"
corpus_bow_file_name = "corpus_bow.mm"
corpus_tfidf_model_file_name = "corpus_tfidf.model"
corpus_tfidf_file_name = "corpus_tfidf.mm"
csr_matrix_file_name = {}
csr_matrix_file_name['TREC8'] = "sparse_matrix_TREC8.npz"
csr_matrix_file_name['TREC7'] = "sparse_matrix_TREC7.npz"
meta_data_file_name = {}
meta_data_file_name['TREC8'] = "metadata_TREC8.pickle"
meta_data_file_name['TREC7'] = "metadata_TREC7.pickle"


topic_original_qrels_doc_list_file_name = "topic_original_qrels_doc_list"
topic_original_qrels_filename = "per_topic_original_qrels"
topic_original_qrels_in_doc_index_filename = "per_topic_original_qrels_in_doc_index"
topic_budget_from_official_qrels_file_name = "per_topic_budget_from_official_qrels"

topic_complete_qrels_filename = "per_topic_complete_qrels"

# if we change the number of seeds, please make sure to delete the corresponding per_topic_seed_documents_IS.pickle under result directory
# otherwise it will always use the previuos seed documents
number_of_seeds = 10
# train_per_centage[0] -> 0 means we only consider the seed documents collected
train_per_centage = np.arange(0.0,1.1,0.1)

# classifier related info for Logistic Regression
small_data_solver = 'liblinear'
small_data_C_parameter = 100000000

large_data_solver = 'saga'
large_data_C_parameter = 1

# number_of_features in a TF-IDF vector
dictionary_features_number = 15000

#plot related info
color_list = ['-b','-r','-g']
marker_list = ['^','o','D']
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

class relevance(object):
    def __init__(self, priority, index):
        self.priority = priority
        self.index = index
        return
    def __cmp__(self, other):
        return -cmp(self.priority, other.priority)


def calculate_entropy(a,b):
    first_part = 0.0
    second_part = 0.0
    if a>0:
        first_part = a * log(a, 2)
    if b>0:
        second_part = b* log(b, 2)
    return (-1)*(first_part+second_part)

def update_dict_key_wise(dict1, dict2, operation = 'add'):
    for k, v in dict2.iteritems():
        if k not in dict1:
            dict1[k] = v
        else:
            if operation == 'add':
                dict1[k] = dict1[k] + v
    return copy.deepcopy(dict1)



# List1 is the ground truth and list 2 is the predicted list
# testing tau_ap implementation
'''
x1 = [1,2,3,4,5,6]
x2 = [2,3,1,4,6,5]
x3 = [6,5,4,3,2,1]

print tau_ap_mine(x1,x2)
print tau_ap_mine(x2,x1)
print tau_ap_mine(x1,x1)
print tau_ap_mine(x1,x3)
print tau_ap_mine(x3,x1)
'''

def tau_ap_mine(list1, list2):

    list1 = sort_dictionary_by_value_return_keys(list1)
    list2 = sort_dictionary_by_value_return_keys(list2)

    length = len(list2)
    c = [0] * length

    for i in xrange(1, len(list2)):
        index_of_element_in_i_list2_in_ground_list = list1.index(list2[i])
        for j in xrange(0,i):
            index_of_element_j_in_list2_in_ground_list = list1.index(list2[j])
            if index_of_element_in_i_list2_in_ground_list > index_of_element_j_in_list2_in_ground_list:
                c[i-1] += 1

    summation = 0
    for i in xrange(1,length):
        summation = summation + (1.0 * c[i-1] / (i))

    p = float(summation)/ (length-1)

    return 2 * p - 1.0

# input is a list of values
def sort_dictionary_by_value_return_keys(list1):
    # convert list into a dictionary
    i = 1
    dict1 = {}
    for elem in list1:
        dict1[i] = elem
        i = i + 1

    # tuples of (key, values)
    listofTuples = sorted(dict1.items(), key=lambda x: -x[1])

    list_of_keys = []
    for element in listofTuples:
        list_of_keys.append(element[0])

    return list_of_keys



a = [0.9, 0.1, 1.0, 0.5]
print sort_dictionary_by_value_return_keys(a)
#print manual_run_list['gov2']
#print manual_run_list['TREC7']
datasource_name_list = ['TREC8', 'TREC7', 'gov2', 'WT2013', 'WT2014']

for datasouce_name in datasource_name_list:
    print datasouce_name, len(manual_run_list[datasouce_name])

for datasouce_name in datasource_name_list:
    for run in manual_run_list[datasouce_name]:
        if run not in systemNameList[datasouce_name]:
            print datasouce_name, run


group_list_without_manual_runs = {}
for datasouce_name in datasource_name_list:
    group_list_without_manual_runs[datasouce_name] = generate_manural_run_free_group_list(group_list[datasouce_name], datasouce_name)
    print datasouce_name, len(group_list[datasouce_name]), len(group_list_without_manual_runs[datasouce_name])
    for i, run_list in group_list_without_manual_runs[datasouce_name].iteritems():
        print datasouce_name, i, run_list


#update dataframe for TREC7 and TREC8
def update_dataframe(dataframe, datasource):
    a = list(xrange(2,11,1))
    b = list(xrange(15, len(dataframe), 3))
    print len(dataframe)
    c = sorted(a + b)
    c.append(len(dataframe))

    new_dataframe = []
    for elem in c:
        #print elem, elem - 2
        new_dataframe.append(dataframe[elem - 2]) # elem -2 because dataframe starts at 2

    return new_dataframe, c



#update dataframe for TREC7 and TREC8
def update_dataframe_test_run(dataframe, datasource):
    a = list(xrange(2,11,1))
    b = list(xrange(15, len(dataframe), 3))
    print len(dataframe)
    c = sorted(a + b)
    c.append(len(dataframe))
    print c
    new_dataframe = []
    for elem in c:
        print elem, elem - 2
        new_dataframe.append(dataframe[elem - 2]) # elem -2 because dataframe starts at 2

    return new_dataframe, c

print count_number_of_relevant(qrelAddress['TREC7'])

for datasource, listofsystem in sorted(systemNameList.iteritems()):
    print datasouce_name_toacronym[datasource], "\!\! & ", len(group_list[datasource]), "\!\! & ", len(manual_run_list[datasource]),  "\!\! & ", len(listofsystem) - len(manual_run_list[datasource]), "\!\! & ", pool_depth[datasource], " \\\\"