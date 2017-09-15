#!/usr/bin/env python
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller, KnowledgeIntegrator
import MySQLdb
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import threading, sys, os, time, json, traceback, pandas, numpy

class NELController:
    def __init__(self, facts_file):
        with open(facts_file) as fd:
            json_data = json.load(fd)

        #print(type(json_data['Classifiers']))
        #print(json_data['Classifiers'])
        classifiers = json_data['Classifiers']
        for classifier in classifiers:
            self.createClassifier(classifier)


    def createClassifier(self, class_dict):
        #print (class_dict)
        classifier_name = class_dict['Classifier_Name']
        data_source = class_dict['Data_Source']
        algorithm = class_dict['Algorithm']
        target = class_dict['Target']
        features = class_dict['Features']
        print(classifier_name)
        print(data_source)
        print(algorithm)
        print(target)
        print(features)

    def parseFeatures(self, feature_string):
        pass

def main():
    filename = "config/facts.json"
    NELController(filename)

if __name__ == '__main__':
    main()
