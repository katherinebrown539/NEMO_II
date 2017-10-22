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
        print("In testKI...")
        #train, holdout = train_test_split(self.data, test_size=0.1)
        split_ind = int(0.1*len(self.data))
        print("split index: " + str(split_ind))
        holdout = self.data[:split_ind]
        train = self.data[split_ind:]
        train.index = list(range(len(train)))
        holdout.index = list(range(len(holdout)))

        predictions = []
        for classifier in self.level1_classifiers:
            predictions.append([])

        #shuffle data, will do this later
        #split training data into k folds
        kf = KFold(n_splits=k, random_state=random_seed, shuffle=False)#will shuffle data manually above
        #fit first stage models on k-1 folds
        x, y = self.splitDataIntoXY(train)
        for train_index, test_index in kf.split(train):
            #print("TRAIN:", train_index, "TEST:", test_index)
            training, testing = train.iloc[train_index], train.iloc[test_index]
            train_x_train, train_y_train = self.splitDataIntoXY(training)
            train_x_test, test_y_test = self.splitDataIntoXY(testing)
            i = 0
            for classifier in self.level1_classifiers:
                #classifier.fit(train_x_train, train_y_train)
                predictions[i].extend(classifier.predict(train_x_test))
                i = i+1
        columns = []
        for classifier in self.level1_classifiers:
            columns.append(classifier.name)

        #out-of-folds <- first stage models predict kth fold
        predictions = pandas.DataFrame(predictions)
        predictions = predictions.transpose()
        predictions.columns = columns
        predictions_x = pandas.concat(objs=[x,predictions], axis=1)
        predictions_y = y
        print("PREDICTIONS:")
        print(predictions)

        #train stacker
        self.stacking_classifier.fit(predictions_x, predictions_y)
        #now predict holdout
        x, y = self.splitDataIntoXY(holdout)
        holdout_predictions = []
        for classifier in self.level1_classifiers:
            holdout_predictions.append([])
        for classifier in self.level1_classifiers:
            predictions[i].extend(classifier.predict(x))
            i = i+1
        predictions = pandas.DataFrame(predictions)
        predictions = predictions.transpose()
        predictions.columns = columns
        predictions_x = pandas.concat(objs=[x,predictions], axis=1)
        predictions_y = y
        stacking_predictions = self.stacking_classifier(predictions_x)
        results = {}
        #GET RIGHT SCORES
        # accuracy
        results['Accuracy'] = sklearn.metrics.accuracy_score(y, stacking_predictions)
        # precision recall f1 support
        results['Precision'], results['Recall'], results['F1'], results['Support'] = sklearn.metrics.precision_recall_fscore_support(y, stacking_predictions)
        # roc
        results['ROC'] = sklearn.metrics.roc_curve(y, stacking_predictions)
        results['ROC_AUC'] = sklearn.metrics.roc_auc_score(y, stacking_predictions)
        return results

    def splitDataIntoXY(self, data):
        x = self.kb.X
        y = self.kb.Y
        while(x.count(y) > 0):
            x.remove(y)
        print("x = " + str(x))
        X = data[self.kb.X]
        print("X:")
        print(X)
        Y = data[[self.kb.Y]]
        print("Y:")
        print(Y)
        return(X,Y)
