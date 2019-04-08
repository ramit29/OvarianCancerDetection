import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import tree
import graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#K_Means_200_Normal_Late
cancer = open("cluster_features/K_Means_200_Normal_Early.csv","r")
next(cancer)
cancer_data = cancer.readlines()
cancer_main=[]
for i in cancer_data:
    i=i.strip("\n").split(",")
    if i[1] == 'Early':
        i[1] = 1
    else:
        i[1] = 0
    cancer_main.append(i)
cancer_x=[]
for i in cancer_main:
    cancer_x.append(i[2:])
cancer_met=[]
for i in cancer_x:
    cancer_met.append([float(j) for j in i])
cancer_target=[]
for i in cancer_main:
    cancer_target.append(int(i[1]))


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


print(len(rfecv.support_))
feats = cancer_met.T.tolist()
print(len(feats))
optimised_feats = []
for i,j in zip(rfecv.support_,feats):
    if i == True:
        optimised_feats.append(j)
optimised_feats=np.array(optimised_feats).T

kf = StratifiedKFold(n_splits=10)
classifiers = [
    AdaBoostClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
    MLPClassifier(),
    DecisionTreeClassifier(),
    SVC(kernel='linear', C=1),
    LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=3000)
    ]

classifier_name = ['Adaptive Boosting','Gaussian Naive Bayes', 'K-Neighbors Classifier', 'Neural Network', 'Decision Tree', 'Support Vector Machine', 'Logistic Regression']
cancer_target = np.asarray(cancer_target)

classifiers_accuracy = {}
for clf,name in zip(classifiers,classifier_name):
    accuracy=[]
    f1 = []
    precision = []
    recall =[]
    for train_indices, test_indices in kf.split(optimised_feats,cancer_target):

        clf.fit(optimised_feats[train_indices], cancer_target[train_indices])
        predict = clf.predict(optimised_feats[test_indices])
        f1.append(f1_score(predict, cancer_target[test_indices]))
        recall.append(recall_score(predict, cancer_target[test_indices]))
        precision.append(precision_score(predict, cancer_target[test_indices]))
        accuracy.append(accuracy_score(predict, cancer_target[test_indices]))
    mean_accuracy=np.mean(accuracy)

    mean_f1 = np.mean(f1)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    classifiers_accuracy[name] = mean_accuracy,mean_f1,mean_precision,mean_recall


output=open("accuracy/{}_accuracy.csv".format(cancer.name.strip(".csv").split("/")[1]),"w+")
#output=open("accuracy/{}_accuracy.csv".format(cancer.name.strip(".csv")),"w+")
output.write("Classifier,Accuracy,f1,precision,recall\n")
for i,j in classifiers_accuracy.items():
    output.write("{},{},{},{},{}\n".format(i,j[0],j[1],j[2],j[3]))
