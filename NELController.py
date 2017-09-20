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

class NELController:
    def __init__(self, facts_file, config_file):
        with open(facts_file) as fd:
            json_data = json.load(fd)

        self.NEMO = NEMO.NEMO(config_file)
        self.classifiers = []
        classifiers = json_data['Classifiers']
        for classifier in classifiers:
            created_classifier = self.createClassifier(classifier)
            self.classifiers.append(created_classifier)
            print created_classifier

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

        new_kb = kb #WILL NEED TO FIX THIS!!

        new_kb.X = x
        new_kb.Y = y
        ml = ML_Controller.ML_Controller(kb, algorithm)
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
