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
        features = self.parseFeatures(features)
        #kb = copy.deepcopy(self.NEMO.getDataSource(data_source))



    def parseFeatures(self, feature_string):
        pass

def main():
    facts = "config/facts.json"
    NELController(facts, "config/config.json")

if __name__ == '__main__':
    main()
