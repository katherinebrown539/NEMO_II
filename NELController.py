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

        #print (json_data)
        for classifier in json_data:
            print (type(classifier))


def main():
    filename = "config/facts.json"
    NELController(filename)

if __name__ == '__main__':
    main()
