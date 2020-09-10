import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from global_definition import *
from topic_description import TRECTopics

base_address = "/work/04549/mustaf/maverick/data/TREC/"
plotAddress = base_address + "plot/"

#per topic number of documents
dataset = 'TREC7'
datapath = base_address + dataset + "/"
filename = 'topicInfo'
qrel_TREC7 = qrelAddress[dataset]
trecData = TRECTopics(dataset, start_topic[dataset], end_topic[dataset])
topicInfo = trecData.qrelsReader(qrelAddress[dataset], datapath, filename)

TREC7_total = []
TREC7 = []

for topicId in topicInfo.iterkeys():
    docInfo = topicInfo[topicId]
    TREC7_total.append(len(docInfo))
    relevant_count = 0
    for docId, docLabel in docInfo.iteritems():
        if docLabel == 1:
            relevant_count = relevant_count + 1

    TREC7.append(relevant_count)

print TREC7_total
print TREC7

print sum(TREC7_total), sum(TREC7), ((sum(TREC7)*1.0)/sum(TREC7_total))*100


WT2014_total = [274, 286, 214, 245, 404, 287, 212, 261, 247, 328, 281, 324, 324, 312, 233, 351, 238, 331, 316, 250, 266, 367, 245, 316, 222, 319, 281, 229, 285, 401, 225, 334, 220, 277, 284, 320, 175, 259, 395, 259, 225, 369, 378, 423, 215, 250, 228, 249, 241, 457]
WT2013_total = [322, 231, 384, 323, 354, 420, 232, 249, 185, 336, 423, 435, 201, 305, 367, 387, 326, 341, 513, 473, 368, 175, 265, 351, 487, 404, 246, 157, 160, 172, 171, 292, 391, 298, 185, 175, 161, 169, 249, 396, 285, 202, 342, 174, 145, 202, 421, 194, 223, 207]
gov2_total = [317, 581, 1403, 549, 483, 993, 444, 530, 551, 470, 604, 525, 441, 448, 725, 532, 529, 544, 868, 444, 622, 543, 544, 725, 883, 632, 381, 684, 976, 835, 653, 1048, 584, 705, 665, 662, 642, 556, 560, 521, 698, 593, 606, 639, 756, 601, 663, 703, 612, 711]
TREC8_total = [2739, 1652, 1046, 1533, 1539, 1420, 1545, 1269, 1476, 1524, 2056, 2113, 1662, 1306, 1309, 1235, 2992, 1748, 1670, 1136, 1763, 2121, 1308, 1747, 1546, 2095, 1528, 1645, 1156, 1709, 1431, 2503, 2162, 1676, 1836, 1949, 1796, 1798, 2015, 1830, 1554, 2679, 2230, 1691, 1404, 2020, 1588, 2408, 1451, 1221]

# per topic number of relevant document
TREC8 = [300, 80, 21, 142, 38, 13, 68, 118, 22, 65, 27, 123, 69, 39, 136, 42, 75, 116, 16, 33, 83, 152, 21, 171, 162, 202, 50, 118, 11, 6, 130, 28, 13, 347, 117, 180, 72, 173, 219, 54, 17, 94, 102, 17, 62, 162, 16, 46, 67, 293]
gov2= [128, 162, 12, 190, 6, 77, 146, 87, 171, 16, 66, 130, 117, 157, 134, 23, 111, 88, 146, 261, 64, 42, 126, 365, 113, 172, 52, 134, 24, 29, 34, 571, 73, 344, 84, 50, 71, 75, 237, 90, 150, 97, 63, 25, 87, 63, 86, 208, 52, 84]
WT2013 = [211, 1, 135, 120, 60, 160, 82, 39, 13, 37, 180, 18, 82, 214, 119, 255, 152, 137, 60, 65, 217, 91, 150, 41, 3, 72, 50, 44, 88, 49, 58, 46, 75, 167, 14, 121, 10, 30, 75, 190, 100, 25, 77, 17, 9, 74, 24, 29, 44, 20]
WT2014 = [181, 122, 146, 133, 5, 89, 105, 70, 112, 24, 21, 210, 153, 70, 21, 241, 85, 131, 27, 66, 15, 227, 40, 100, 24, 199, 62, 4, 98, 184, 105, 202, 96, 229, 215, 123, 47, 143, 47, 30, 22, 308, 121, 261, 85, 156, 164, 151, 133, 62]

prevalenceTREC7 = []
prevalenceTREC8 = []
prevalencegov2 = []
prevalenceWT2013 = []
prevalenceWT2014 = []

i = 0
for i in xrange(0, len(TREC8)):
    val = (TREC8[i]*100.0)/TREC8_total[i]
    prevalenceTREC8.append(val)
print prevalenceTREC8

i = 0
for i in xrange(0, len(TREC7)):
    val = (TREC7[i]*100.0)/TREC7_total[i]
    prevalenceTREC7.append(val)
print prevalenceTREC7

i = 0
for i in xrange(0, len(gov2)):
    val = (gov2[i]*100.0)/gov2_total[i]
    prevalencegov2.append(val)
print prevalencegov2

i = 0
for i in xrange(0, len(WT2013)):
    val = (WT2013[i]*100.0)/WT2013_total[i]
    prevalenceWT2013.append(val)
print prevalenceWT2013


i = 0
for i in xrange(0, len(WT2014)):
    val = (WT2014[i]*100.0)/WT2014_total[i]
    prevalenceWT2014.append(val)
print prevalenceWT2014




data = [prevalenceWT2014, prevalenceWT2013, prevalencegov2, prevalenceTREC8, prevalenceTREC7]
plt.boxplot(data)

plt.grid()
plt.xlabel("TREC Track",size = 16)
plt.ylabel("% of relevant documents per topic", size = 16)
plt.ylim(0,100)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
#plt.show()

plt.tight_layout()
# plt.show()
## add patch_artist=True option to ax.boxplot()
## to get fill color
#exit(0)

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data)
bp = ax.boxplot(data, patch_artist=True)

## change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set( facecolor = '#1b9e77' )

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='r', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)

plt.xticks([1, 2, 3, 4, 5], ['WT\'14', 'WT\'13', 'TB\'06', 'Adhoc\'99', 'Adhoc\'98'])
plt.savefig(plotAddress +'perTopicPrevalence.pdf', format='pdf')