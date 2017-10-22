from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller
from collections import deque
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split, KFold
import pandas, MySQLdb, threading, sys, os, time, random

class AutoKnowledgeIntegrator:
    def __init__(self, kb, level1_classifiers, stacking_classifier=None, use_features=False):
        self.kb = kb
        self.level1_classifiers = level1_classifiers
        if stacking_classifier is None or stacking_classifier == "Logistic Regression":
            self.algorithm_name = "KI_LogisticRegression"
            self.stacking_classifier = LogisticRegression()
        elif stacking_classifier == "Decision Tree":
            self.algorithm_name = "KI_DecisionTree"
            self.stacking_classifier = DecisionTreeClassifier()
        elif stacking_classifier == "SVM":
            self.algorithm_name = "KI_SVM"
            self.stacking_classifier = SVC()
        self.algorithm_id = "KI"+str(random.randint(1,101))
        self.use_features = use_features
        self.data = self.kb.getData()
        print("DATA")
        print(self.data)

    def testKI(self, k = 10, random_seed = None):
        train, holdout = train_test_split(self.data, test_size=0.1)
        predictions = []
        i = 0
        for classifier in self.level1_classifiers:
            predictions[i] = []
            i = i+1

        #shuffle data, will do this later
        #split training data into k folds
        kf = KFold(n_splits=k, random_state=random_seed, shuffle=False)#will shuffle data manually above
        #fit first stage models on k-1 folds
        train_x, train_y = self.splitDataIntoXY(train)
        for train_index, test_index in kf.split(X):
            train_x_train, train_x_test = train_x[train_index], train_x[test_index]
            train_y_train, train_y_test = train_y[train_index], train_y[test_index]
            i = 0
            for classifier in self.level1_classifiers:
                classifier.fit(train_x_train, train_y_train)
                predictions[i].append(classifier.predict(train_x_test))
        columns = []
        for classifier in self.level1_classifiers:
            columns.append(classifier.name)
        predictions = pandas.DataFrame(predictions, columns)
        print("PREDICTIONS:")
        print(predictions)
        #out-of-folds <- first stage models predict kth fold
        #split out of folds into kp folds
        #fit stacker on kp-1 folds and predict pth fold
        #fit first stage models on training set without holdout
        #X <- predict the holdout of the trained first stage models
        #use X in second stage model
        #cv error is error on second stage prediction on X

    def splitDataIntoXY(self, data):
        x = self.kb.X
        y = self.kb.Y
        while(x.count(y) > 0):
            x.remove(y)
        print("x = " + str(x))
        X = data[self.kb.X]
        print("X:")
        print(self.X)
        Y = data[[self.kb.Y]]
        print("Y:")
        print(self.y)
        return(X,Y)
