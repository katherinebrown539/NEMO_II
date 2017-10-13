#!/usr/bin/env python
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller, KnowledgeIntegrator
import NEMO
import MySQLdb
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import threading, sys, os, time, json, traceback, pandas, numpy
import copy
from ConstraintLanguage import ConstraintLanguage

class NELController:
    def __init__(self, facts_file, config_file):
        with open(facts_file) as fd:
            json_data = json.load(fd)

        self.NEMO = NEMO.NEMO(config_file)
        self.NEMO.resetAlgorithmResults()
        self.classifiers = []
        classifiers = json_data['Classifiers']
        self.createClassifiers(classifiers)
        self.NEMO.printAlgorithmResults()
        #parse constraints
        self.constraints = []
        self.blankets = []
        self.parseConstraints(json_data['Constraints'])
        self.generateMarkovBlanket()
        self.runBlanketsInKI()
    #will need to generalize for other data sets......
    def runBlanketsInKI(self):
        self.runTraumaBlanketsInKI()

    def runTraumaBlanketsInKI(self):
        iss_blanket = self.blankets[0]
        iss_kb = None
        clses = []
        num_folds = 10
        random_seed = 0
        for classifier in iss_blanket['CLASSIFIERS_THAT_INFLUENCE']:
            if classifier['Class'] == 'ISS16':
                iss_kb = classifier['Classifier'].kb
            clses.append(classifier['Classifier'])
        KI = KnowledgeIntegrator.KnowledgeIntegrator(iss_kb, clses, stacking_classifier='Decision Tree', other_predictions=None, use_features=False)
        data = iss_kb.getData()
        shuffled_data = shuffle(data)
        splits = numpy.array_split(shuffled_data, num_folds)
        results = []
        results.append(KI_Lung.testKI(splits, num_folds, random_seed))
        for r in results:
            print(r)
            
    def runORNLBlanketsInKI(self):
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
                    if classifier['Class'] == 'LUNG':
                            lung_kb = classifier['Classifier'].kb
                    lung_classifiers.append(classifier['Classifier'])
            elif b['RIGHT_MEMBER'] == 'BREAST':
                breast_blanket = b
                for classifier in b['CLASSIFIERS_THAT_INFLUENCE']:
                    if classifier['Class'] == 'BREAST':
                            breast_kb = classifier['Classifier'].kb
                    breast_classifiers.append(classifier['Classifier'])
        KI_Lung = KnowledgeIntegrator.KnowledgeIntegrator(lung_kb, lung_classifiers, stacking_classifier='Decision Tree', other_predictions=None, use_features=False)
        KI_Breast = KnowledgeIntegrator.KnowledgeIntegrator(breast_kb, breast_classifiers, stacking_classifier='Decision Tree', other_predictions=None, use_features=False)
        #run KIs
        data = lung_kb.getData()
        shuffled_data = shuffle(data)
        splits = numpy.array_split(shuffled_data, 10)
        num_folds = 10
        random_seed = 0
        results = []
        results.append(KI_Lung.testKI(splits, num_folds, random_seed))
        data = breast_kb.getData()
        shuffled_data = shuffle(data)
        splits = numpy.array_split(shuffled_data, 10)
        results.append(KI_Breast.testKI(splits, num_folds, random_seed))
        for r in results:
            print r

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
        print "BLANKETS:"
        for b in self.blankets:
            print b

    def createClassifiers(self, classifiers):
        for classifier in classifiers:
            created_classifier = self.createClassifier(classifier)
            self.classifiers.append(created_classifier)
            #run classifiers
            created_classifier['Classifier'].runAlgorithm()
            created_classifier['Classifier'].updateDatabaseWithResults()

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
        new_kb.X = x
        new_kb.Y = y
        print("Algorithm: " + algorithm)
        ml = ML_Controller.ML_Controller(new_kb, algorithm)
        ml.createModel()
        d =  {"Classifier_Name": classifier_name, "Class": target, "Classifier": ml}
        #print d
        return d

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
    facts = "config/facts.json"
    NELController(facts, "config/config.json")

if __name__ == '__main__':
    main()
