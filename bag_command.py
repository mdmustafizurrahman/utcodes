import sys
import os

dataset_list = ['TREC8']
protocol_list = ['CAL']
al_classifier = ['LR']
start_topic = 401
end_topic = 451
pooled_budget = 1
use_original_qrels = 0
varied_qrels_directory_number = 4
current_directory = os.getcwd()
time = '00:30:00'
node = 1

s=''
batch_script = "#!/bin/sh\n"
variation = 0
for datasource in dataset_list: # 1
    for protocol in protocol_list: #4
        for classifier in al_classifier:
            for topic in xrange(start_topic, end_topic):
                shellcommand = '#!/bin/sh\n' \
                               '#SBATCH -J '+datasource+protocol+str(topic) + '# job name\n' \
                               '#SBATCH -o '+ datasource+protocol+str(topic) + '.o%j    # output and error file name (%j expands to jobID)\n' \
                               '#SBATCH -N 1             # total number of nodes\n' \
                               '#SBATCH -n '+str(node)+ '     # total number of mpi tasks requested\n' \
                               '#SBATCH -p normal           # queue (partition) -- normal, development, etc.\n' \
                               '#SBATCH -t '+str(time)+ '       # run time (hh:mm:ss) - 1.0 hours\n' \
                               '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                               '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                               '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                               '\n\nmodule load python\n\n'

                #print "python "+ "tfidfLoader.py " + datasource +" "+ protocol+" IS NR all "+classifier + " "+str(topic)
                s = shellcommand + "python "+ "tfidfLoader.py " + datasource +" "+ protocol+" IS NR all "+classifier + " "+str(topic)+" "+str(pooled_budget)+" "+str(use_original_qrels)+" "+str(varied_qrels_directory_number)
                filename = current_directory + '/' + datasource+protocol+str(topic) + '.slurm'
                print filename
                text_file = open(filename, "w")
                text_file.write(s + "\n")
                text_file.close()
                batch_script = batch_script + 'sbatch '+ filename +"\n"
                variation = variation + 1



print "Number of variations:" + str(variation)

filename1 = current_directory + '/batch_command.sh'
print filename1
text_file = open(filename1, "w")
text_file.write(batch_script)
text_file.close()

