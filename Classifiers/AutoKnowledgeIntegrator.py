from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller
from collections import deque
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, precision_recall_fscore_support,roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import pandas, MySQLdb, threading, sys, os, time, random, numpy

class AutoKnowledgeIntegrator:
    def __init__(self, kb, level1_classifiers, stacking_classifier=None, use_features=False):
        #print("Init KI")
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
        self.name = self.kb.name + "_" + self.kb.Y + "_KI_"+stacking_classifier
        #print("DATA")
        #print(self.data)

    def testKI(self, k = 10, random_seed = None):
        #print("In testKI...")
        self.data = shuffle(self.data)
        self.data.index = list(range(len(self.data)))
        results = {}
        results['Accuracy'] = []
        results['Precision'] = []
        results['Recall'] = []
        results['F1'] = []
        results['Support'] = []
        results['ROC'] = []
        results['ROC_AUC'] = []
        results['Confusion_Matrix'] = []
        kf = KFold(n_splits=k, random_state=random_seed, shuffle=False)
        for train_index, test_index in kf.split(self.data):
            train, holdout = self.data.iloc[train_index], self.data.iloc[test_index]
            train.index = list(range(len(train)))
            holdout.index = list(range(len(holdout)))
            temp_results = self.cv_step(train, holdout, k, random_seed)
            results['Accuracy'].append(temp_results['Accuracy'])
            results['Precision'].append(temp_results['Precision'])
            results['Recall'].append(temp_results['Recall'])
            results['F1'].append(temp_results['F1'])
            results['Support'].append(temp_results['Support'])
            results['ROC'].append(temp_results['ROC'])
            results['ROC_AUC'].append(temp_results['ROC_AUC'])
            results['Confusion_Matrix'].append(temp_results['Confusion_Matrix'])
        results['Accuracy'] = numpy.mean(results['Accuracy'])
        results['Precision'] = numpy.mean(results['Precision'])
        results['Recall'] = numpy.mean(results['Recall'])
        results['F1'] = numpy.mean(results['F1'])
        results['Support'] = numpy.mean(results['Support'])
        results['ROC'] = numpy.mean(results['ROC'])
        results['ROC_AUC'] = numpy.mean(results['ROC_AUC'])
        print(results)
        return results

    def predict(self, x):
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
        predictions = pandas.DataFrame(predictions)
        predictions = predictions.transpose()
        predictions.columns = columns
        predictions_x = pandas.concat(objs=[x,predictions], axis=1)


    def cv_step(self, train, holdout, k, random_seed):
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
        stacking_predictions = self.stacking_classifier.predict(predictions_x)
        return stacking_predictions
        #rint("PREDICTIONS:")
        #print(predictions)

        #train stacker
        self.stacking_classifier.fit(predictions_x, predictions_y)
        #now predict holdout
        x, y = self.splitDataIntoXY(holdout)
        # print("X for Holdout:")
        # print(x)
        # print("Y for Holdout:")
        # print(y)

        holdout_predictions = []
        for classifier in self.level1_classifiers:
            holdout_predictions.append([])
        i = 0
        for classifier in self.level1_classifiers:
            holdout_predictions[i].extend(classifier.predict(x))
            i = i+1
        holdout_predictions = pandas.DataFrame(holdout_predictions)
        holdout_predictions = holdout_predictions.transpose()
        holdout_predictions.columns = columns
        predictions_x = pandas.concat(objs=[x,holdout_predictions], axis=1)
        predictions_y = y
        # print("PREDICTIONS_X:")
        # print(predictions_x)
        stacking_predictions = self.stacking_classifier.predict(predictions_x)
        results = {}
        #GET RIGHT SCORES
        # accuracy
        results['Accuracy'] = accuracy_score(y, stacking_predictions)
        # precision recall f1 support
        results['Precision'] = precision_score(y, stacking_predictions)
        results['Recall'] = recall_score(y, stacking_predictions)
        results['F1'] = f1_score(y, stacking_predictions)
        prec,rec,f,results['Support'] = precision_recall_fscore_support(y, stacking_predictions)
        # roc
        results['ROC'] = roc_curve(y, stacking_predictions)
        results['ROC_AUC'] = roc_auc_score(y, stacking_predictions)
        results['Confusion_Matrix'] = confusion_matrix(y, stacking_predictions)
        #sprint(results)
        return results

    def splitDataIntoXY(self, data):
        x = self.kb.X
        y = self.kb.Y
        while(x.count(y) > 0):
            x.remove(y)
        #print("x = " + str(x))
        X = data[self.kb.X]
        #print("X:")
        #print(X)
        Y = data[[self.kb.Y]]
        #print("Y:")
        #print(Y)
        return(X,Y)
