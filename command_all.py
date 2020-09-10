import sys
import os
from global_definition import *

#dataset_list = ['TREC7', 'TREC8', 'gov2', 'WT2014', 'WT2013']
dataset_list = ['gov2']


# for leave one out setup
group_start_number = {}
group_start_number['TREC8'] = [2, 21, 30, 37]
group_start_number['TREC7'] = [2, 21, 30, 37]
group_start_number['gov2'] = [2]
group_start_number['WT2013'] = [2]
group_start_number['WT2014'] = [2]


group_end_number = {}
group_end_number['TREC8'] = [20, 29, 36, 40]
group_end_number['TREC7'] = [20, 29, 36, 41]
group_end_number['gov2'] = [20]
group_end_number['WT2013'] = [16]
group_end_number['WT2014'] = [12]


'''
group_start_number = {}
group_start_number['TREC8'] = [2, 21]
group_start_number['TREC7'] = [2, 21]
group_start_number['gov2'] = [2]
group_start_number['WT2013'] = [2]
group_start_number['WT2014'] = [2]


group_end_number = {}
group_end_number['TREC8'] = [20, 30]
group_end_number['TREC7'] = [20, 30]
group_end_number['gov2'] = [15]
group_end_number['WT2013'] = [12]
group_end_number['WT2014'] = [9]
'''

group_start_number_auto = {}
group_start_number_auto['TREC8'] = [2, 21, 30]
group_start_number_auto['TREC7'] = [2, 21, 30]
group_start_number_auto['gov2'] = [2]
group_start_number_auto['WT2013'] = [2]
group_start_number_auto['WT2014'] = [2]


group_end_number_auto = {}
group_end_number_auto['TREC8'] = [20, 29, 34]
group_end_number_auto['TREC7'] = [20, 29, 36]
group_end_number_auto['gov2'] = [20]
group_end_number_auto['WT2013'] = [14]
group_end_number_auto['WT2014'] = [12]


datasource_time = {}
datasource_time['TREC8'] = 50
datasource_time['TREC7'] = 50
datasource_time['gov2'] = 50
datasource_time['WT2013'] = 40
datasource_time['WT2014'] = 40

datasource_hour = {}
datasource_hour['TREC8'] = '00'
datasource_hour['TREC7'] = '00'
datasource_hour['gov2'] = '00'
datasource_hour['WT2013'] = '00'
datasource_hour['WT2014'] = '00'

#rank_metric_list = ['map', 'P.10', 'infAP']
rank_metric_list = ['map']

excluded_list = [0,1,2,3,4]
onlyautomatic_list = [0]

current_directory = os.getcwd()


s=''
j='#!/bin/sh\n'

variation = 0
for datasource in dataset_list: # 1
    datasource_variation = 1
    shellcommand = '#!/bin/sh\n' \
                   '#SBATCH -J tau_matt# job name\n' \
                   '#SBATCH -o tau_matt.o%j       # output and error file name (%j expands to jobID)\n' \
                   '#SBATCH -N 1              # total number of nodes\n' \
                   '#SBATCH -n 1             # total number of mpi tasks requested\n' \
                   '#SBATCH -p normal     # queue (partition) -- normal, development, etc.\n' \
                   '#SBATCH -t '+str(datasource_hour[datasource])+':'+ str(datasource_time[datasource]) +':00        # run time (hh:mm:ss) - 1.0 hours\n' \
                   '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                   '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                   '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                   '\n\nmodule load python\n\n'
    for rank_metric in rank_metric_list:
        for elem in excluded_list:
            group_start = {}
            group_end = {}
            for pool_depth_to_use in pool_depth_variation[datasource]:
                for onlyautomatic in onlyautomatic_list:
                    if onlyautomatic == 1:
                        group_start = group_start_number_auto
                        group_end = group_end_number_auto
                    if onlyautomatic == 0:
                        group_start = group_start_number
                        group_end = group_end_number
                    for start_pos, end_pos in zip(group_start[datasource], group_end[datasource]):
                        print "python "+ "rankTester.py " + datasource +" CAL IS NR qrels LR "+ str(start_topic[datasource]) + " "+ str(end_topic[datasource]) + " " + rank_metric + " " + str(elem) + " " + str(start_pos) + " " + str(end_pos) + " " + str(onlyautomatic) + " " + str(pool_depth_to_use)
                        s = shellcommand + "python "+ "rankTester.py " + datasource +" CAL IS NR qrels LR "+ str(start_topic[datasource]) + " "+ str(end_topic[datasource]) + " " + rank_metric + " " + str(elem) + " " + str(start_pos) + " " + str(end_pos) + " " + str(onlyautomatic) + " " + str(pool_depth_to_use)
                        filename1 = current_directory + '/batch_command_'+datasource+"_"+str(datasource_variation) +'.slurm'
                        print filename1
                        #print shellcommand
                        text_file = open(filename1, "w")
                        text_file.write(s+"\n")
                        text_file.close()
                        j = j + "sbatch batch_command_"+datasource+"_"+  str(datasource_variation) +".slurm\n"
                        variation = variation + 1
                        datasource_variation = datasource_variation + 1


print "Number of variations:" + str(variation)
filename1 = current_directory + "/all_jobs.batch"
text_file = open(filename1, "w")
text_file.write(j + "\nwait\n")
text_file.close()
