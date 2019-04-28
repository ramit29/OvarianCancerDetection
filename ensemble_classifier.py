import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import GridSearchCV
#K_Means_200_Normal_Late
cancer = open("cluster_features/Normal_Early/Hierarchical_80_Normal_Early.csv","r")
next(cancer)
cancer_data = cancer.readlines()
cancer_main=[]
for i in cancer_data:
    i=i.strip("\n").split(",")
    print(i[1],type(i[1]))
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
X_train, X_test, y_train, y_test = train_test_split(optimised_feats, cancer_target, test_size=0.2, random_state=0)

knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
optimised_feats=np.asarray(optimised_feats)
cancer_target = np.asarray(cancer_target)
params_knn = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)
knn_gs.fit(X_train, y_train)
knn_best = knn_gs.best_estimator_
rf = RandomForestClassifier()
params_rf = {'n_estimators': [50, 100, 200]}
#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)
#fit model to training data
rf_gs.fit(X_train, y_train)
#save best model
rf_best = rf_gs.best_estimator_
log_reg = LogisticRegression()
#fit the model to the training data
log_reg.fit(X_train, y_train)

#check best n_estimators value
#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')

#fit model to training data
accuracy=[]
for train_indices, test_indices in kf.split(optimised_feats,cancer_target):

    ensemble.fit(optimised_feats[train_indices], cancer_target[train_indices])
    accuracy.append(ensemble.score(optimised_feats[test_indices], cancer_target[test_indices]))

print(np.mean(accuracy))
