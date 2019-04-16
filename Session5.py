############## Classification#############

import pandas as pd
import numpy as np


###reading data
# missValue=['?']
data2= pd.read_csv("EC-H1-train.csv")
data1= pd.read_csv("EC-H1-test.csv")
##replacing
# bmedian = data2['Bare Nuclei'].median()
# data2['Bare Nuclei'].fillna(bmedian,inplace=True)

X_train=data2.values[:,1:]
y_train=data2.values[:,0]
X_test=data1.values[:,1:]
y_test=data1.values[:,0]


###########Logistic Regression#####
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logregmodel = logreg.fit(X_train, y_train)
y_pred = logregmodel.predict(X_test)



########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred , y_test )
print(cnf_matrix)

#####Metrics
print("logistic Regression_Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))





##########Naive Bayes #########
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()
GNBmodel = GNB.fit(X_train, y_train)
y_pred = GNBmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred,y_test )
print(cnf_matrix)

#####Metrics
print("GaussianNB_Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F_measure:",metrics.f1_score(y_test, y_pred))



######QDA ###########
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
QDA= QuadraticDiscriminantAnalysis()
QDAmodel = QDA.fit(X_train, y_train)
y_pred = QDAmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#####Metrics
print("QDA_Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




######LDA ###########
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA= LinearDiscriminantAnalysis()
LDAmodel = LDA.fit(X_train, y_train)
y_pred = LDAmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred ,y_test )
print(cnf_matrix)

#####Metrics
print("LDA_Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))



###### Classification with Linear Regression######
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LRmodel = LR.fit(X_train, y_train)
y_regpred = LRmodel.predict(X_test)
y_pred= [1 if x>=0.4 else 0 for x in y_regpred]

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred , y_test)
print(cnf_matrix)

#####Metrics
print("Linear Regression_Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# from sklearn.model_selection import LeaveOneOut
# X = X_train
# y = y_train
# loo = LeaveOneOut()
# loo.get_n_splits(X)
#
#
# for train_index, test_index in loo.split(X):
#    # print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    # print(X_train, X_test, y_train, y_test)

1+1