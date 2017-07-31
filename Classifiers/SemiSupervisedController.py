from Classifiers import ML_Controller, KnowledgeIntegrator
from KnowledgeBase import KnowledgeBase
from collections import deque
from NEMO import NEMO
import pandas
from pandas import DataFrame
import numpy
from sklearn.utils import shuffle
import pandas.io.sql as psql
import MySQLdb
import sys
import os
import time
import random
import json

class SemiSupervisedController:
	def __init__(self, classifiers, kb, split_method, k):
		self.classifiers = classifiers
		self.ki = None
		self.kb = kb
		if split_method in ['kRandomSplit']:
			self.split_method = split_method
		else:
			self.split_method = 'kRandomSplit'
			

		self.k = k
		self.training_instances = None #KATIE, BE CAREFUL WITH THIS!!!!!
		self.test_instances = None
	
	def split(self, seed = None, data = None):
		if self.split_method == 'kRandomSplit':
			self.kRandomSplit(seed, data)
	
	def kRandomSplit(self, seed, data=None):
		if data is None:
			stmt = "select * from DATA"
			data = pandas.read_sql_query(stmt, self.kb.db)
		
		if isinstance(self.k, float):
			self.k = int(self.k * len(data.index))
		
		shuffled_data = shuffle(data, random_state=seed)
		self.training_instances = shuffled_data.iloc[0:self.k]
		self.test_instances = shuffled_data.iloc[self.k:len(shuffled_data.index)]
	
		#self.training_instances = shuffled_data.head(n=self.k)
		#tail_num = len(shuffled_data.index) - self.k
		#self.test_instances = shuffled_data.tail(n=tail_num)
		
	def trainClassifiers(self, stacking_classifier, other_predictions, use_features):
		x_train = self.training_instances[self.kb.X]
		y_train = self.training_instances[self.kb.Y]
		x_test = self.test_instances[self.kb.X]
		y_test = self.test_instances[self.kb.Y]
		
		
		for classifier in self.classifiers:
			classifier.createModelPreSplit(x_train, x_test, y_train, y_test)
			#classifier.fit(x_train, y_train)
		self.ki = KnowledgeIntegrator.KnowledgeIntegrator( self.kb, self.classifiers, stacking_classifier, other_predictions, use_features)
		
	
	def testClassifiers(self, metric, seed, num_folds):
		res = {}
		for classifier in self.classifiers:
			cls_res = classifier.runAlgorithm(self.test_instances[self.kb.X], self.test_instances[self.kb.Y])
			res[classifier.getName()] = classifier.algorithm.results.get(metric)
			
			splits = numpy.array_split(self.test_instances, 10)
			#ki_res = self.testKI(self.ki,splits)
			ki_res = self.ki.testKI(splits, num_folds, seed)

			res[self.ki.getName()] = ki_res.get(metric)
		
		return res
	
	def kMeansPlusPlus(self):
		pass
	
	def kMedoids(self):
		pass
	