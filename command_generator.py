import sys
import os

dataset_list = ['WT2014']
protocol_list = ['CAL','SAL','SPL']
topic_sampling_protocol = [3] # 1 means MAB, 0 mean deterministic, 2 means oracle, 3 means smart oracle
run_number_list = [0]

current_directory = os.getcwd()

shellcommand = '#!/bin/sh\n' \
               '#SBATCH -J topic_distribution_mab# job name\n' \
               '#SBATCH -o topic_distribution_mab.o%j       # output and error file name (%j expands to jobID)\n' \
               '#SBATCH -N 1              # total number of nodes\n' \
               '#SBATCH -n 3             # total number of mpi tasks requested\n' \
               '#SBATCH -p gpu     # queue (partition) -- normal, development, etc.\n' \
               '#SBATCH -t 6:59:59        # run time (hh:mm:ss) - 1.0 hours\n' \
               '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
               '#SBATCH --mail-type=begin  # email me when the job starts\n' \
               '#SBATCH --mail-type=end    # email me when the job finishes\n' \
               '\n\nmodule load python\n\n'

s=''
variation = 0
for datasource in dataset_list: # 1
    for protocol in protocol_list: #4
        for sampling_protocol in topic_sampling_protocol:
            for run_number in run_number_list:
                    #if sampling_protocol == 0 and run_number>0:
                    #    continue
                    print "python "+ "topic_distribution_mab.py " + datasource +" "+ protocol+" "+str(sampling_protocol) +" "+str(run_number)   +" &"
                    s = s + "python "+ "topic_distribution_mab.py " + datasource +" "+ protocol+" "+ str(sampling_protocol)+" "+str(run_number)  +" &\n"
                    variation = variation + 1


print "number of variations:", variation
filename1 = current_directory + '/batch_command_'+dataset_list[0] +'_'+str(run_number_list[0])+'.slurm'
print filename1
print shellcommand
text_file = open(filename1, "w")
text_file.write(shellcommand+s+"\nwait\n")
text_file.close()

print "Number of variations:" + str(variation)
