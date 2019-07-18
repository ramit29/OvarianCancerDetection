"""from sklearn.metrics import r2_score
batchoneclean=open('clean_batchone_dupl.csv','r')
batchtwoclean=open('clean_batchtwo_dupl.csv','r')

batchoneclean=batchoneclean.readlines()
batchone_clean={}
for i in batchoneclean:
    i=i.strip("\n").split(",")
    #print(i[0])
    val=[float(m) for m in i[1:]]
    batchone_clean[i[0]]=val
#print(len(batchone_clean))
batchtwoclean=batchtwoclean.readlines()
batchtwo_clean={}
for i in batchtwoclean:
    i=i.strip("\n").split(",")
    #print(i[0])
    val=[float(m) for m in i[1:]]
    batchtwo_clean[i[0]]=val
#print(len(batchtwo_clean))
for i,j in zip(batchone_clean,batchtwo_clean):
    print(i,j)
print(len(batchone_clean),len(batchtwo_clean))
r2_clean=[]
for i,j in zip(batchone_clean,batchtwo_clean):
    r2_clean.append(r2_score(i,j))
import matplotlib.pyplot as plt
import numpy as np
r2_clean=np.array(r2_clean)
print(r2_clean)
range=np.arange(1,38)




combatoneclean=open('combat_batchone_dupl.csv','r')
combattwoclean=open('combat_batchtwo_dupl.csv','r')

combatoneclean=combatoneclean.readlines()
combatone_clean=[]
for i in combatoneclean:
    i=i.strip("\n").split(",")
    #print(i[0])
    val=[float(m) for m in i[1:]]
    combatone_clean.append(val)
#print(len(combatone_clean))
combattwoclean=combattwoclean.readlines()
combattwo_clean=[]
for i in combattwoclean:
    i=i.strip("\n").split(",")
    #print(i[0])
    val=[float(m) for m in i[1:]]
    combattwo_clean.append(val)
#print(len(combattwo_clean))
r2_combat=[]
for i,j in zip(combatone_clean,combattwo_clean):
    r2_combat.append(r2_score(i,j))
r2_combat=np.array(r2_combat)
print(r2_combat)
plt.plot(range,r2_clean)
plt.plot(range,r2_combat)
plt.show()
"""
import subprocess
subprocess.call('pwd')
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("cleandata_july16.csv")
#data_norm = pd.read_csv("check_combat/OneDrive_1_7-17-2019/BatchLocationCorrected-master_input_B1-2.csv")
data_norm = pd.read_csv("neg_batchloccombat.csv")

DuplicatesBatches = pd.read_csv("Duplicates.csv", dtype=object)
batch1 = DuplicatesBatches['Batch 1'].astype(str)
batch2 = DuplicatesBatches['Batch 2'].astype(str)
batch3 = DuplicatesBatches['Batch 3'].astype(str)
batch4 = DuplicatesBatches['Batch 4'].astype(str)



#Original data
data = data.transpose()

#Make first row, column names
data.columns = data.iloc[0].apply(str)
data = data.drop(data.index[0:4])

#Make sample names consistant
data.columns = data.columns.str.replace('HOC_', '')
data.columns = data.columns.str.replace('_0618', '')
data.columns = data.columns.str.replace('_0728', '')


#ComBat corrected data
data_norm = data_norm.transpose()

#Make first row, column names
data_norm.columns = data_norm.iloc[0].apply(str)
data_norm = data_norm.drop(data_norm.index[0:4])

#Make sample names consistant
data_norm.columns = data_norm.columns.str.replace('HOC_', '')
data_norm.columns = data_norm.columns.str.replace('_0618', '')
data_norm.columns = data_norm.columns.str.replace('_0728', '')
print(data_norm.columns)

coefficient_of_dermination_prebatch12 = [r2_score(data[batch1[i]], data[batch2[i]]) for i in range(len(batch1))]
coefficient_of_dermination_postbatch12 = [r2_score(data_norm[batch1[i]], data_norm[batch2[i]]) for i in range(len(batch1))]

duplicates = range(len(batch1))

plt.plot(duplicates, coefficient_of_dermination_prebatch12, 'r', label="r^2 pre batch correction")
plt.plot(duplicates, coefficient_of_dermination_postbatch12, 'b', label="r^2 post batch correction")

plt.title('R-squared values for 37 duplicate samples between batches 1 and 2 before and after batch correction')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
