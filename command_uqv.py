import sys
import os

dataset_list = ['WT2014', 'WT2013']
system_name_list = ['user-Atire-MC2.uqv.run','user-Indri-BM.uqv.run', 'user-Indri-LM.uqv.run', 'user-Terrier-DFR.uqv.run', 'user-Terrier-PLC.uqv.run']
buffer_size = 100
measure = "map"

current_directory = os.getcwd()
shellcommand = '#!/bin/sh\n' \
               '#SBATCH -J uqv_analysis# job name\n' \
               '#SBATCH -o uqv_analysis.o%j       # output and error file name (%j expands to jobID)\n' \
               '#SBATCH -N 1              # total number of nodes\n' \
               '#SBATCH -n 1             # total number of mpi tasks requested\n' \
               '#SBATCH -p normal     # queue (partition) -- normal, development, etc.\n' \
               '#SBATCH -t 00:25:00        # run time (hh:mm:ss) - 1.0 hours\n' \
               '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
               '#SBATCH --mail-type=begin  # email me when the job starts\n' \
               '#SBATCH --mail-type=end    # email me when the job finishes\n' \
               '\n\nmodule load python2\n\n'

s=''
j='#!/bin/sh\n'

variation = 0
for datasource in dataset_list: # 1
    for system_name in system_name_list:
        job_name = datasource + "_" + system_name
        shellcommand = '#!/bin/sh\n' \
                       '#SBATCH -J uqv_analysis_'+ job_name + '# job name\n' \
                       '#SBATCH -o uqv_analysis.'+ job_name +'o%j       # output and error file name (%j expands to jobID)\n' \
                       '#SBATCH -N 1              # total number of nodes\n' \
                       '#SBATCH -n 1             # total number of mpi tasks requested\n' \
                       '#SBATCH -p normal     # queue (partition) -- normal, development, etc.\n' \
                       '#SBATCH -t 00:25:00        # run time (hh:mm:ss) - 1.0 hours\n' \
                       '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                       '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                       '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                       '\n\nmodule load python2\n\n'


        s= ''
        print "python "+ "uqvCode_v1.py " + datasource +" "+ system_name+" " + measure +" "+str(buffer_size)
        s = s + "python "+ "uqvCode_v1.py " + datasource +" "+ system_name+" " + measure +" "+ str(buffer_size) + "\n"
        variation = variation + 1

        print "number of variations:", variation
        filename1 = current_directory + '/uqv_command_'+datasource +'_'+ system_name +'.slurm'
        print filename1
        #print shellcommand
        text_file = open(filename1, "w")
        text_file.write(shellcommand+s+"\nwait\n")
        text_file.close()
        j = j + "sbatch "+ 'uqv_command_'+datasource +'_'+ system_name +'.slurm\n'


print "Number of variations:" + str(variation)
filename1 = current_directory + "/all_jobs.batch"
text_file = open(filename1, "w")
text_file.write(j + "\nwait\n")
text_file.close()
