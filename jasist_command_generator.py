import sys
import os

#dataset_list = ['TREC8', 'gov2','WT2013','WT2014']
#dataset_list = ['WT2013']
#dataset_list = ['WT2014']
#dataset_list = ['gov2']
dataset_list = ['TREC8']

protocol_list = ['CAL','SAL','SPL']
use_ranker_list = ['False', 'True']
iter_sampling_list = ['False', 'True']
current_directory = os.getcwd()


s=''
variation = 0
for datasource in dataset_list: # 1
    for protocol in protocol_list: #4
        for use_ranker in use_ranker_list:
            for iter_sampling in iter_sampling_list:
                    if iter_sampling == 'False' and use_ranker == 'True':
                        print "Skipping", use_ranker, iter_sampling
                        continue
                    time = ''
                    node = ''
                    if datasource=='WT2013' or datasource == 'WT2014':
                        time = '00:04:00' # actula 2:15s
                        node = '9'
                    elif datasource == 'gov2':
                        time = '00:07:00'  # actula 5:18s
                        node = '9'
                    else:
                        time = '0:11:00' # actual 9:19s
                        node = '9'

                    shellcommand = '#!/bin/sh\n' \
                                   '#SBATCH -J jasist# job name\n' \
                                   '#SBATCH -o jasist.o%j    # output and error file name (%j expands to jobID)\n' \
                                   '#SBATCH -N 1             # total number of nodes\n' \
                                   '#SBATCH -n '+str(node)+ '     # total number of mpi tasks requested\n' \
                                   '#SBATCH -p gpu           # queue (partition) -- normal, development, etc.\n' \
                                   '#SBATCH -t '+str(time)+ '       # run time (hh:mm:ss) - 1.0 hours\n' \
                                   '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                                   '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                                   '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                                   '\n\nmodule load python\n\n'

                    print "python "+ "jasist.py " + datasource +" "+ protocol+" "+str(use_ranker) +" "+str(iter_sampling) + " &"
                    s = s + "python " + "jasist.py " + datasource + " " + protocol + " " + str(
                            use_ranker) + " " + str(iter_sampling) + " &\n"


                    variation = variation + 1


print "number of variations:", variation
filename1 = current_directory + '/jasist_command_'+dataset_list[0]+'.slurm'
print filename1
print shellcommand
text_file = open(filename1, "w")
text_file.write(shellcommand+s+"\nwait\n")
text_file.close()

print "Number of variations:" + str(variation)
