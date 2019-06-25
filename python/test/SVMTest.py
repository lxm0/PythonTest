import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import hamming_loss
if __name__ == '__main__':
    #import our data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    #split the data to  7:3
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    clf = svm.SVC(probability=True)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    #print(clf.predict_proba(X_test))
    # print(clf.score(X_test,y_test))
    # precision = precision_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average='micro')
    hammingloss = hamming_loss(y_test, y_pred)
    print(hammingloss)
    print(precision)
    print(recall)
    # score_rbf = clf_rbf.score(X_test,y_test)
    # print("The score of rbf is : %f"%score_rbf)


