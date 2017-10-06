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
        #print("Printing blankets")
        #for b in self.blankets:
            #print(b)
        #group by Constraint and Right-Class Member
        #markov blanket
        #knowledge integrator
    def generateMarkovBlanket(self):
        #print("CONSTRAINTS:")
        #for c in self.constraints:
            #print(c)

        #create blanket dicts
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

        #cycle through constraints and add to blanket dict
        for i in range(0, len(self.constraints)):
            left_members = []
            right_member = self.constraints[i]['RIGHT_MEMBER']
            for j in range(0, len(self.constraints)):
                if i == j: continue
                if right_member == self.constraints[j]['RIGHT_MEMBER']:
                    left_members.append(self.constraints[j]['RIGHT_MEMBER'])
            for b in self.blankets:
                if b['RIGHT_MEMBER'] == right_member:
                    b['CLASSES_THAT_INFLUENCE'] = left_members
                    break;
        #cycle through classifiers and add to blanket dict
        for classifier in self.classifiers:
            for b in self.blankets:
                #test to see if it matches the RIGHT_MEMBER
                #check to see if matches the LEFT_MEMBER
                if classifier['Class'] == b['RIGHT_MEMBER'] or classifier['Class'] in b['CLASSES_THAT_INFLUENCE']:
                    b['CLASSIFIERS_THAT_INFLUENCE'].append(classifier)

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
            print parsed
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
        #print(x)
        #print(y)

        new_kb = kb.copy() #WILL NEED TO FIX THIS!!
        #new_kb = kb
        new_kb.X = x
        new_kb.Y = y
        print("Algorithm: " + algorithm)
        ml = ML_Controller.ML_Controller(kb, algorithm)
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
