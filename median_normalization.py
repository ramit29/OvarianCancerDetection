import pandas as pd
import numpy as np
batchone=pd.read_csv("positivelog_batchone_dupl.csv",header=None)
batchtwo=pd.read_csv('positivelog_batchtwo_dupl.csv',header=None)
#Dropping column id

batchone = batchone.drop(columns=0)
batchtwo = batchtwo.drop(columns=0)
#sanity check
#print(np.shape(batchone))
batchone_median=batchone.median()

batchtwo_median=batchtwo.median()
#print(batchone_median)
#print(batchtwo_median)
batch_normalized=batchone_median-batchtwo_median
#sanity check
#print(batch_normalized)
#print(batchone)
data=pd.read_csv('positive_log_batchtwo.csv')
dataone=pd.read_csv('positive_log_batchone.csv')
dataone = dataone.transpose()

#Make first row, column names
dataone.columns = dataone.iloc[0].apply(str)
dataone = dataone.drop(dataone.index[0:4])
dataone = dataone.transpose()

data = data.transpose()

#Make first row, column names
data.columns = data.iloc[0].apply(str)
data = data.drop(data.index[0:4])
data = data.transpose()

#print(data)
#print(batchtwo)
#print(batch_normalized)
batch_normalized=np.asarray(batch_normalized)
#print(batchone)
#print(data-batch_normalized)

batchtwo_corrected=dataone - batch_normalized

print(batchtwo_corrected)
print(data)


batchtwo_corrected.to_csv('batchtwocorrected_log.csv')
