#!/usr/bin/env python
import graphviz
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller, KnowledgeIntegrator, AutoKnowledgeIntegrator#, AutoMLController
import NEMO
import MySQLdb
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import threading, sys, os, time, json, traceback, pandas, numpy
import copy
from ConstraintLanguage import ConstraintLanguage
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, precision_recall_fscore_support,roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split, KFold
class NELController:
    def __init__(self, facts_file, config_file, output_file):
        with open(facts_file) as fd:
            json_data = json.load(fd)
        self.results = []
        save_stdout = sys.stdout
        #sys.stdout = open('trash', 'w')
        self.NEMO = NEMO.NEMO(config_file)
        self.output_file = output_file
        self.NEMO.resetAlgorithmResults()
        self.classifiers = []
        classifiers = json_data['Classifiers']
        self.createClassifiers(classifiers)
        self.constraints = []
        self.blankets = []
        self.parseConstraints(json_data['Constraints'])
        self.generateMarkovBlanket()
        #self.runBlanketsInKI()
        self.execute()
        sys.stdout = save_stdout
        # for r in self.results:
        #     print(r)
        self.writeToCSV()
    #will need to generalize for other data sets......
    def runBlanketsInKI(self):
        self.runTraumaBlanketsInKI()
        #self.runORNLBlanketsInKI()

    def execute(self):
        results = []
        i = 0
        data_ = self.classifiers[0].get('Classifier').kb.getData()
        for classifier in self.classifiers:
            r = {}
            r['Classifier'] = classifier['Classifier']
            r['Name'] = classifier['Classifier'].getName()+"_"+classifier['Classifier'].kb.Y
            r['Accuracy'] = []
            r['Precision'] = []
            r['Recall'] = []
            r['F1'] = []
            r['Support'] = []
            r['ROC'] = []
            r['ROC_AUC'] = []
            r['Confusion_Matrix'] = []
            results.append(r)
            i = i+1
        kis = self.generateTraumaKI()
        ki_res = []
        for ki in kis:
            r = {}
            r['Classifier'] = ki
            r['Name'] = ki.getName()
            r['Accuracy'] = []
            r['Precision'] = []
            r['Recall'] = []
            r['F1'] = []
            r['Support'] = []
            r['ROC'] = []
            r['ROC_AUC'] = []
            r['Confusion_Matrix'] = []
            ki_res.append(r)
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(data_):
            for result in results:
                #split into test and training
                #split into x and y
                X,Y = result['Classifier'].kb.splitDataIntoXY()
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                #train classifier
                result['Classifier'].fit(X_train, y_train)
                predict = result['Classifier'].predict(X_test)
                result['Accuracy'].append(accuracy_score(y_test, predict))
                # precision recall f1 support
                result['Precision'].append(precision_score(y_test, predict))
                result['Recall'].append(recall_score(y_test, predict))
                result['F1'].append(f1_score(y_test, predict))
                prec,rec,f,sup = precision_recall_fscore_support(y_test, predict)
                result['Support'].append(sup)# roc
                result['ROC'].append(roc_curve(y_test, predict))
                result['ROC_AUC'].append(roc_auc_score(y_test, predict))
                result['Confusion_Matrix'].append(confusion_matrix(y_test, predict))
                #get test error
                #append to results for this algorithm

                for result in ki_res:
                    result['Classifier'].fitLevel1Classifiers(X_train, y_train)
                    result['Classifier'].fit(X_train, y_train)
                    predict = result['Classifier'].predict(X_test)
                    result['Accuracy'].append(accuracy_score(y_test, predict))
                    # precision recall f1 support
                    result['Precision'].append(precision_score(y_test, predict))
                    result['Recall'].append(recall_score(y_test, predict))
                    result['F1'].append(f1_score(y_test, predict))
                    prec,rec,f,sup = precision_recall_fscore_support(y_test, predict)
                    result['Support'].append(sup)# roc
                    result['ROC'].append(roc_curve(y_test, predict))
                    result['ROC_AUC'].append(roc_auc_score(y_test, predict))
                    result['Confusion_Matrix'].append(confusion_matrix(y_test, predict))
        for result in results:
            result['Accuracy'] = numpy.mean(result['Accuracy'])
            result['Precision'] = numpy.mean(result['Precision'])
            result['Recall'] = numpy.mean(result['Recall'])
            result['F1'] = numpy.mean(result['F1'])
            result['Support'] = numpy.mean(result['Support'])
            result['ROC_AUC'] = numpy.mean(result['ROC_AUC'])
            
        for result in ki_results:
            result['Accuracy'] = numpy.mean(result['Accuracy'])
            result['Precision'] = numpy.mean(result['Precision'])
            result['Recall'] = numpy.mean(result['Recall'])
            result['F1'] = numpy.mean(result['F1'])
            result['Support'] = numpy.mean(result['Support'])
            result['ROC'] = numpy.mean(result['ROC'])
            result['ROC_AUC'] = numpy.mean(result['ROC_AUC'])
            self.results.append(result)
        self.results = results

    def writeToCSV(self):
        #f = open(self.output_file, 'w')
        with open(self.output_file, "w") as f:
            #f.writeline("Algorithm Name, Accuracy, Precision, Recall, F1, Support, ROC, ROC_AUC\n")
            f.write("Algorithm Name, Accuracy, Precision, Recall, F1\n")
            for r in self.results:
                #line = r['Name']+","+str(r['Accuracy'])+","+str(r['Precision'])+","+str(r['Recall'])+","+str(r['F1'])+","+str(r['Support'])+","+str(r['ROC'])+","+str(r['ROC_AUC'])
                line = r['Name']+","+str(r['Accuracy'])+","+str(r['Precision'])+","+str(r['Recall'])+","+str(r['F1'])+"\n"
                f.write(line)

    def printModel(self, model):
        from sklearn import tree
        if model is not None:
            print(str(type(model)))
        dot_data = tree.export_graphviz(model, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("ModelPrintout")

    def runTraumaBlanketsInKI(self):
        kis = []
        ed2or = []
        icuadmit = []
        earlydeath = []
        #Get all classifiers that classify the same thing
        for classifiers in self.classifiers:
            if classifiers['Class'] == 'ED2OR':
                #classifiers['Classifier'].runModel()
                ed2or.append(classifiers['Classifier'])
            elif classifiers['Class'] == 'ICUAdmit':
                #classifiers['Classifier'].runModel()
                icuadmit.append(classifiers['Classifier'])
            elif classifiers['Class'] == 'EarlyDeath':
                #classifiers['Classifier'].runModel()
                earlydeath.append(classifiers['Classifier'])

        ki = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(ed2or[0].kb, ed2or, stacking_classifier='Decision Tree', use_features=False)
        results = ki.testKI()
        results['Name'] = ki.name
        self.results.append(results)
        kis.append(ki)
        ki = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(icuadmit[0].kb, icuadmit, stacking_classifier='Decision Tree', use_features=False)
        kis.append(ki)
        results = ki.testKI()
        results['Name'] = ki.name
        self.results.append(results)
        kis.append(ki)
        ki = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(earlydeath[0].kb, earlydeath, stacking_classifier='Decision Tree', use_features=False)
        kis.append(ki)
        results = ki.testKI()
        results['Name'] = ki.name
        self.results.append(results)
        kis.append(ki)
        for blanket in self.blankets:
            if blanket['RIGHT_MEMBER'] in ['ISS16', 'NeedTC']:
                c = blanket['RIGHT_MEMBER']
                self.executeBlanket(blanket,c, clses_=kis)
                self.executeBlanket(blanket,c, clses_=None)

    def generateTraumaKI(self):
        kis = []
        ed2or = []
        icuadmit = []
        earlydeath = []
        #Get all classifiers that classify the same thing
        for classifiers in self.classifiers:
            if classifiers['Class'] == 'ED2OR':
                #classifiers['Classifier'].runModel()
                ed2or.append(classifiers['Classifier'])
            elif classifiers['Class'] == 'ICUAdmit':
                #classifiers['Classifier'].runModel()
                icuadmit.append(classifiers['Classifier'])
            elif classifiers['Class'] == 'EarlyDeath':
                #classifiers['Classifier'].runModel()
                earlydeath.append(classifiers['Classifier'])
        ki = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(ed2or[0].kb, ed2or, stacking_classifier='Decision Tree', use_features=False)
        kis.append(ki)
        ki = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(icuadmit[0].kb, icuadmit, stacking_classifier='Decision Tree', use_features=False)
        kis.append(ki)
        ki = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(earlydeath[0].kb, earlydeath, stacking_classifier='Decision Tree', use_features=False)
        kis.append(ki)

        for blanket in self.blankets:
            if blanket['RIGHT_MEMBER'] in ['ISS16', 'NeedTC']:
                c = blanket['RIGHT_MEMBER']
                kis.extend(self.executeBlanket(blanket,c, clses_=kis))
                kis.extend(self.executeBlanket(blanket,c, clses_=None))
        return kis

    def runORNLBlanketsInKI(self):
        lung_cls = []
        breast_cls = []
        lung_blanket = None
        breast_blanket = None
        lung_kb = None
        breast_kb = None
        lung_classifiers = []
        breast_classifiers = []
        for b in self.blankets:
            if b['RIGHT_MEMBER'] == 'LUNG':
                lung_blanket = b
                for classifier in b['CLASSIFIERS_THAT_INFLUENCE']:
                    print(classifier['Class'])
                    if classifier['Class'] == 'LUNG':
                        lung_kb = classifier['Classifier'].kb
                        lung_classifiers.append(classifier['Classifier'])
                    elif b['RIGHT_MEMBER'] == 'BREAST':
                        breast_blanket = b
                        for classifier in b['CLASSIFIERS_THAT_INFLUENCE']:
                            print(classifier['Class'])
                            if classifier['Class'] == 'BREAST':
                                breast_kb = classifier['Classifier'].kb
                                breast_classifiers.append(classifier['Classifier'])
                                KI_Lung = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(lung_kb, lung_classifiers, stacking_classifier='Decision Tree', use_features=False)
                                KI_Breast = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(breast_kb, breast_classifiers, stacking_classifier='Decision Tree', use_features=False)
                                #run KIs
                                results = []
                                r = KI_Lung.testKI(splits, num_folds, random_seed)
                                r['Name'] = "ORNL_LUNG_KI"
                                results.append(r)
                                r = KI_Breast.testKI(splits, num_folds, random_seed)
                                r['Name'] = "ORNL_BREAST_KI"
                                results.append(r)

    def executeBlanket(self, blanket, class_, clses_=None, exec_=True):
        kis = []
        kb = None
        if clses_ is None:
            clses = []
        else:
            clses = clses_
        results = []
        num_folds = 10
        random_seed = 0
        for classifier in blanket['CLASSIFIERS_THAT_INFLUENCE']:
            if classifier['Class'] == class_:
                kb = classifier['Classifier'].kb
            if clses_ is None:
                clses.append(classifier['Classifier'])

        KI = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(kb, clses, stacking_classifier='Decision Tree', use_features=False)
        kis.append(KI)
        if(clses_ is not None):
            KI.name = KI.name + "_usingStackers"
        if exec_:
            r = KI.testKI(k = 10, random_seed = random_seed)
            self.results.append(r)
        KI = AutoKnowledgeIntegrator.AutoKnowledgeIntegrator(kb, clses, stacking_classifier='Logistic Regression', use_features=False)
        kis.append(KI)
        if(clses_ is not None):
            KI.name = KI.name + "_usingStackers"
        if exec_:
            r = KI.testKI(k = 10, random_seed = random_seed)
            self.results.append(r)
        return kis


    def generateMarkovBlanket(self):
        #create blanket dicts
        self.blankets = []
        right_members_that_exist = []
        for constraint in self.constraints:
            right_member = constraint['RIGHT_MEMBER']
            #print("right_members_that_exist = " + str(right_members_that_exist))
            #print("current right_member = " + str(right_member))
            if right_member not in right_members_that_exist:
                #print("not in")
                blanket = {}
                blanket['RIGHT_MEMBER'] = right_member
                blanket['CLASSES_THAT_INFLUENCE'] = []
                blanket['CLASSIFIERS_THAT_INFLUENCE'] = []
                right_members_that_exist.append(right_member)
                self.blankets.append(blanket)
        for constraint in self.constraints:
            #print constraint
            to_use = None
            for blanket in self.blankets:
                if blanket['RIGHT_MEMBER'] == constraint['RIGHT_MEMBER']:
                    to_use = blanket
                    break
            to_use['CLASSES_THAT_INFLUENCE'].append(constraint['LEFT_MEMBER'])
        for classifier in self.classifiers:
            for blanket in self.blankets:
                if (classifier['Class'] in blanket['CLASSES_THAT_INFLUENCE']) or (classifier['Class'] == blanket['RIGHT_MEMBER']):
                    blanket['CLASSIFIERS_THAT_INFLUENCE'].append(classifier)
        print ("BLANKETS:")
        for b in self.blankets:
            print (b)

    def createClassifiers(self, classifiers):
        for classifier in classifiers:
            created_classifier = self.createClassifier(classifier)
            self.classifiers.append(created_classifier)
            #run classifiers
            # created_classifier['Classifier'].runAlgorithm()
            # created_classifier['Classifier'].updateDatabaseWithResults()
            # r = created_classifier['Classifier'].algorithm.results
            # r['Name'] = created_classifier['Classifier'].name
            # self.results.append(r)

    def parseConstraints(self, constraint_data):
        #constraint_data = json_data['Constraints']
        print(constraint_data)
        stuff = []
        for c in constraint_data:
            parser = ConstraintLanguage()
            #print c
            parsed = parser.parse(c['Relationship'])
            #print parsed
            stuff.append(parser.parse(c['Relationship']))
        self.constraints = stuff

    def createClassifier(self, class_dict):
        #print (class_dict)
        classifier_name = class_dict['Classifier_Name']
        data_source = class_dict['Data_Source']
        algorithm = class_dict['Algorithm']
        target = class_dict['Target']
        features = class_dict['Features']
        #features = self.parseFeatures(features)
        kb = self.NEMO.getDataSourceFromName(data_source) #will need copy constructor for KnowledgeBase
        all_feats = kb.X
        all_feats.append(kb.Y)
        #print("Got KB")
        x,y = self.parseFeatures(features, target, all_feats)
        print(x)
        print(y)

        new_kb = kb.copy() #WILL NEED TO FIX THIS!!
        #new_kb = kb
        new_kb.setNewXY(x,y)
        #new_kb.X = x
        #new_kb.Y = y
        print("Algorithm: " + algorithm)
        ml = ML_Controller.ML_Controller(new_kb, algorithm)
        #ml = AutoMLController.AutoMLController(new_kb, algorithm)
        ml.createModel()
        d =  {"Classifier_Name": classifier_name, "Class": target, "Classifier": ml}
        #print d
        return d

    def runModel(self, classifier):
        self.results.append(classifier.runModel())
    #update comment
    def parseFeatures(self, feature_string, target, all_features):
        #print("Feature String: " + feature_string)
        #print("Target: " + target)
        if feature_string[0] == '{':
            #pre-specified features
            #print("Case 1")
            feature_string = feature_string.strip('{}')
            features = feature_string.split(',')
            print("Features: " + str(features))
            print("Target: " + str(target))
            return(features,target)
        elif len(feature_string) >= 4 and feature_string[4] == '-':
            #print("Case 2") #all minus case
            feature_string = feature_string[6:]
            feature_string = feature_string.strip('{}')
            features = feature_string.split(',')
            while(all_features.count(target)):
                all_features.remove(target)
            for f in features:
                while(all_features.count(f)):
                    all_features.remove(f)
            return(all_features,target)
        elif feature_string == 'ALL':
            #print("Case 3") #all features
            #print(all_features.count(target))
            while(all_features.count(target) > 0):
                all_features.remove(target)
            return(all_features,target)
        else:
            print("Invalid feature string")


def main():
    output_file = None
    if(len(sys.argv) > 1):
        output_file = sys.argv[1]
    else:
        output_file = "results.csv"
    facts = "config/facts.json"
    NELController(facts, "config/config.json", output_file)

if __name__ == '__main__':
    main()
