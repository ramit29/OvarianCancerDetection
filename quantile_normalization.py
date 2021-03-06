import numpy as np

import pandas





#cancer = open("cluster_features/Normal_Early/Hierarchical_200_Normal_Early.csv","r")
cancer = open("cat_transpose.csv","r")
next(cancer)

#next(cancer)
cancer = cancer.readlines()
colnames=cancer[0]

colnames=colnames.strip("\n").split(",")
colu=[]
for i in colnames:
    #print(i.strip('""'))
    colu.append(i.strip('""'))
cancer= cancer[1:]#print(len(cancer))
cancer_main=[]
for i in cancer:
    i=i.strip("\n")
    i=i.strip('"').split(",")
    #print(i[3])
    if i[1] == 'Cancer':
        #print(i[1])
        i[1] = 'Cancer'
    else:
        i[1] = 'Normal'
    i[0]=i[0].strip('"')
    i[2]=i[2].strip('"')
    cancer_main.append(i)
cancer_x=[]
for i in cancer_main:
    cancer_x.append(i[3:])
cancer_meta = []
for i in cancer_main:
    cancer_meta.append(i[:3])

cancer_met=[]
for i in cancer_x:
    cancer_met.append([float(j.strip('"')) for j in i])
cancer_target=[]
for i in cancer_main:
    cancer_target.append(i[1])

df = pandas.DataFrame(cancer_met)
df1 = pandas.DataFrame(cancer_meta)
#print(cancer_met)
rank_mean = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
cancer_met = df.rank(method='min').stack().astype(int).map(rank_mean).unstack()

colmeta=colu[:3]
colval=colnames[3:]
df1.columns=colmeta
print(df1)
print(cancer_met)
df1.to_csv("meta.csv",index_label='Number')
cancer_met.to_csv("quantile.csv",index_label='Number')
