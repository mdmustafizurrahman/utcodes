from nltk.corpus import stopwords

'''
Step 0:
=======================
Change the variable of IndriParameterGenerator accordinlgy
 
Step 1:
========================
 ./buildIndex parameter
 ./dumpindex /export/home/u16/nahid/data/IndriIndex/TREC8 stats
kstem_add_table_entry: Duplicate word emeritus will be ignored.
Repository statistics:
documents:	528155
unique terms:	664573
total terms:	253367449
fields:		text 

Step 2:
========================
Run: /export/home/u16/nahid/code/script.sh
'''

parameter_file = "/export/home/u16/nahid/tools/indri-5.13/parameter"
index_address = "/export/home/u16/nahid/data/IndriIndex/TREC8"
document_collection = "/export/home/u16/nahid/data/TREC/TIPSTER/collection/TREC8"
memory = '4G'
storeDocs = 'true'
TREC_type = 'trectext' # can be trecweb for gov2, WT2013, WT2014

parameter_string_start = "<parameters>\n" \
                   "    <index>" + index_address + "</index>\n" \
                   "    <memory>" + memory + "</memory>\n" \
                   "    <storeDocs>true</storeDocs>\n" \
                   "    <corpus>\n" \
                   "      <path>"+ document_collection +"</path>\n" \
                   "    <class>"+  TREC_type +"</class>\n" \
                   "    </corpus>\n" \
                   "    <stemmer><name>krovetz</name></stemmer>\n" \
                   "    <field>\n" \
                   "    <name>TEXT</name>\n" \
                   "    </field>\n" \
                   "    <stopper>\n"



parameter_string_end = "</stopper>\n</parameters>"
stopwords_list = stopwords.words("english")
word_start = "    <word>"
word_end = "</word>"

all_stopwords = ""
for w in stopwords_list:
    word_tag = word_start + w + word_end + "\n"
    all_stopwords = all_stopwords + word_tag

parameter_string = parameter_string_start + all_stopwords + parameter_string_end

print parameter_string

text_file = open(parameter_file, "w")
text_file.write(parameter_string)
text_file.close()
