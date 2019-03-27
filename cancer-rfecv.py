import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import tree
import graphviz
#K_Means_200_Normal_Late
cancer = open("K_Means_200_Normal_Late.csv","r")
next(cancer)
cancer = cancer.readlines()
cancer_main=[]
for i in cancer:
    i=i.strip("\n").split(",")
    if i[1] == 'Late':
        i[1] = 1
    else:
        i[1] = 0
    cancer_main.append(i)
cancer_x=[]
for i in cancer_main:
    cancer_x.append(i[3:])
cancer_met=[]
for i in cancer_x:
    cancer_met.append([float(j) for j in i])
cancer_target=[]
for i in cancer_main:
    cancer_target.append(i[1])
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10),
              scoring='accuracy')
scaling = MinMaxScaler(feature_range=(-1,1)).fit(cancer_met)
cancer_met = scaling.transform(cancer_met)
rfecv.fit(cancer_met, cancer_target)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
