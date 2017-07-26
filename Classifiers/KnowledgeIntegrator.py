from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller
from collections import deque
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas
import NEMO
import MySQLdb
import threading
import sys
import os
import time

class KnowledgeIntegrator:
	def __init__(self, kb, level1_classifiers, stacking_classifier=None):
		self.kb = kb
		self.level1_classifiers = level1_classifiers
		if stacking_classifier is None or stacking_classifier == "Logistic Regression":
			self.name = "KI_LogisticRegression"
			self.stacking_classifier = LogisticRegression()
		elif stacking_classifier == "Decision Tree":
			self.stacking_classifier = DecisionTreeClassifier()
			self.name = "KI_DecisionTree"
		elif stacking_classifier == "SVM":
			self.stacking_classifier = SVC()
			self.name = "KI_SVM"
		self.meta_data_set = []
		
		#self.keys.append(self.kb.Y)
			
	def trainLevelOneModels(self, fold):
		xtrain,	xtest, ytrain, ytest = fold
		for classifier in self.level1_classifiers:
			classifier.createModelPreSplit(xtrain, xtest, ytrain, ytest)
		
	def evaluateLevelOneModels(self, x):
		to_return = [] #holds list of predictions and truth
		for classifier in self.level1_classifiers:
			curr = classifier.predict(x)
			# print curr
			# print ""
			to_return.append(curr)
			
		return to_return
		
	def trainAndCreateMetaDataSet(self, folds):
		self.resetKeys()
		names = []
		names = self.keys
		names.append(self.kb.Y)
		self.meta_data_set = []
		for fold in folds:
			xtrain,	xtest, ytrain, ytest = fold
			self.trainLevelOneModels(fold)
			predictions = self.evaluateLevelOneModels(xtest)
			predictions.append(ytest.values)
			predictions = pandas.DataFrame(predictions).T
			predictions.columns = names
			self.meta_data_set.append(predictions)
		self.meta_data_set = pandas.concat(self.meta_data_set)
		#print self.meta_data_set
		
	def createMetaDataSet(self, folds):
		self.resetKeys()
		names = self.keys
		names.append(self.kb.Y)
		set = []
		for fold in folds:
			xtrain,	xtest, ytrain, ytest = fold
			predictions = self.evaluateLevelOneModels(xtest)
			predictions.append(ytest.values)
			predictions = pandas.DataFrame(predictions).T
			predictions.columns = names
			self.meta_data_set.append(predictions)
		self.meta_data_set = pandas.DataFrame(set)
		#print self.meta_data_set
		
	def trainMetaModel(self, data=None):
		if data is None:
			data = self.meta_data_set
		#print data
		x,y = self.splitMetaIntoXY(data)
		
		x = self.transform(x)
		self.stacking_classifier.fit(x, y)
		
	def transform(self, x):
		le = preprocessing.LabelEncoder()
		#LabelEncoder()
		le.fit([ "c34.0", "c34.1", "c34.2", "c34.3", "c34.9", "c50.1", "c50.2", "c50.3", "c50.4", "c50.5", "c50.8", "c50.9"])
		new_x = []
		for column in x:
			#print column
			column = x[column]
			#print column
			new_x.append(le.transform(column))
		x = pandas.DataFrame(new_x)
		return x.T
		
	def runModel(self, data):
		x,y = self.splitIntoXY(data)
	
		predictions = self.evaluateLevelOneModels(x)
		predictions.append(y.values)
		
		self.meta_data_set.append(predictions)	
		self.meta_data_set = pandas.DataFrame(self.meta_data_set)
		# print "Meta DATA Set"
		# print self.meta_data_set
		
		x,y = self.splitMetaIntoXY(self.meta_data_set)
		x = self.transform(x)
		predictions = self.stacking_classifier.predict(x)
		av = 'micro'
		
		accuracy = accuracy_score(y,predictions)
		precision = precision_score(y,predictions, average=av)
		recall = recall_score(y, predictions, average=av)
		f1 = f1_score(y,predictions, average=av)
		cm = confusion_matrix(y,predictions)
		
		to_return =  {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "CM": cm}
		return to_return
			
	def crossValidateMetaModel(self, k):
		pass
		
	def getName(self):
		return self.name
		
	def splitMetaIntoXY(self, data):
		cols = deque(data.columns.tolist())
		y_name = cols.pop()
		x_names = list(cols)
		y = data[y_name] #need to change to reflect varying data...
		x = data[x_names]

		return(x,y)
		
	def splitIntoXY(self, data):
		#print data
		#print(data.columns.tolist())
		y = data[self.kb.Y] #need to change to reflect varying data...
		#print y
		x = data[self.kb.X]
		#print x
		return (x,y)
		
	def resetKeys(self):
		self.keys = []
		for classifier in self.level1_classifiers:
			key = classifier.getName() + "_" + classifier.getID()
			self.keys.append(key)