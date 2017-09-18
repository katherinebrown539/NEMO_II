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

        classifiers = json_data['Classifiers']
        for classifier in classifiers:
            created_classifier = self.createClassifier(classifier)


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
        self.parseFeatures(features, target, all_feats)

    def parseFeatures(self, feature_string, target, all):
        print(feature_string)
        print(feature_string[0])
        #case1: {}
            #make list of strings
        #case2: ALL - {}
            #get all column names and remove
        #case3: ALL
            #get all but target

def main():
    facts = "config/facts.json"
    NELController(facts, "config/config.json")

if __name__ == '__main__':
    main()
