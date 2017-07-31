from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller
from collections import deque
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas
#import NEMO
import MySQLdb
import threading
import sys
import os
import time
#git test for git
class KnowledgeIntegrator:
	def __init__(self, kb, level1_classifiers, stacking_classifier=None, other_predictions=None, use_features=False):
		self.kb = kb
		self.level1_classifiers = level1_classifiers
		if stacking_classifier is None or stacking_classifier == "Logistic Regression":
			self.algorithm_name = "KI_LogisticRegression"
			self.stacking_classifier = LogisticRegression()
		elif stacking_classifier == "Decision Tree":
			self.stacking_classifier = DecisionTreeClassifier()
			self.algorithm_name = "KI_DecisionTree"
		elif stacking_classifier == "SVM":
			self.stacking_classifier = SVC()
			self.algorithm_name = "KI_SVM"
		self.meta_data_set = []
		self.other_predictions = other_predictions
		#self.keys.append(self.kb.Y)
		self.algorithm_id = "111111111"
		self.use_features = use_features
			
	def trainLevelOneModels(self, fold):
		xtrain,	xtest, ytrain, ytest = fold
		if self.other_predictions is not None:
				split = self.splitIntoAttributesOther(xtrain)
				xtrain = split[0]
				other_train = split[1]
				split = self.splitIntoAttributesOther(xtest)
				xtest = split[0]
				other_test = split[1]
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
			other_train = None
			other_test = None
			#strip the other_predictions -> other_predictions_train, other_predictions_test
			if self.other_predictions is not None:
				split = self.splitIntoAttributesOther(xtrain)
				xtrain = split[0]
				other_train = split[1]
				split = self.splitIntoAttributesOther(xtest)
				xtest = split[0]
				other_test = split[1]
			self.trainLevelOneModels(fold)
			predictions = self.evaluateLevelOneModels(xtest)
			#append the other_predictions_test
			if self.other_predictions is not None:
				#print other_test
				predictions.append(other_test.values)
				#print pandas.DataFrame(predictions).T
			predictions.append(ytest.values)
			predictions = pandas.DataFrame(predictions).T
			#print predictions
			predictions.columns = names
			
			if self.use_features:
				predictions.index = xtest.index
				#print predictions
				predictions = predictions.merge(xtest, left_index=True, right_index=True)
				# print predictions
				
			self.meta_data_set.append(predictions)
		self.meta_data_set = pandas.concat(self.meta_data_set)

	def createMetaDataSet(self, folds):
		self.resetKeys()
		names = self.keys
		names.append(self.kb.Y)
		set = []
		for fold in folds:
			xtrain,	xtest, ytrain, ytest = fold
			other_train = None
			other_test = None
			#strip other_predictions into other_predictions_train, other_predictions_test
			if self.other_predictions is not None:
				split = self.splitIntoAttributesOther(xtrain)
				xtrain = split[0]
				other_train = split[1]
				split = self.splitIntoAttributesOther(xtest)
				xtest = split[0]
				other_test = split[1]
			predictions = self.evaluateLevelOneModels(xtest)
			if self.other_predictions is not None:
				predictions.append(other_test.values)
			predictions.append(ytest.values)
			predictions = pandas.DataFrame(predictions).T
			predictions.columns = names
			
			if self.use_features:
				predictions.index = xtest.index
				#print predictions
				predictions = predictions.merge(xtest, left_index=True, right_index=True)
				# print predictions
				
			self.meta_data_set.append(predictions)
		self.meta_data_set = pandas.DataFrame(self.meta_data_set)
		
		
	def trainMetaModel(self, data=None):
		if data is None:
			data = self.meta_data_set
		# print data
		# print "Data"
		# print data
		x,y,features = self.splitMetaIntoXY(data)
		#print x
		x = self.transform(x)
		if self.use_features:
			x.index = features.index	
			x = x.merge(features, right_index = True, left_index = True)
			print "x"
			print x
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
		print len(data)
		print "data"
		#print data
		x,y = self.splitIntoXY(data)
		#strip the other_predictions
		other = None
		if self.other_predictions is not None:
			split = self.splitIntoAttributesOther(x)
			x = split[0]
			other = split[1]
		predictions = self.evaluateLevelOneModels(x)
		#append other_predictions
		if self.other_predictions is not None:
				predictions.append(other.values)
		predictions.append(y.values)
		self.resetKeys()
		names = self.keys
		names.append(self.kb.Y)
		print len(self.meta_data_set)
		self.meta_data_set = pandas.DataFrame(predictions).T
		self.meta_data_set.columns = names
		
		
		
		if self.use_features:
			self.meta_data_set.index = x.index
			self.meta_data_set = self.meta_data_set.merge(x, left_index=True, right_index=True)
			print self.meta_data_set
		
		
		x,y,features = self.splitMetaIntoXY(self.meta_data_set)
		print "y"
		print y
		print "features"
		print features
		x = self.transform(x)
		print "x"
		print x
		
		if self.use_features:
			x.index = features.index	
			x = x.merge(features, right_index = True, left_index = True)
		print x	
		predictions = self.stacking_classifier.predict(x)
		av = 'micro'
		
		accuracy = accuracy_score(y,predictions)
		precision = precision_score(y,predictions, average=av)
		recall = recall_score(y, predictions, average=av)
		f1 = f1_score(y,predictions, average=av)
		cm = confusion_matrix(y,predictions)
		
		#to_return =  {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "Confusion_Matrix": cm}
		to_return =  {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "Confusion_Matrix": cm}
		return to_return
	
	def testKI(self, splits, num_folds, random_seed):
		print "in test KI"
		print self.meta_data_set
		self.meta_data_set = []
		holdout = splits.pop()
		remain = pandas.concat(splits)
		folded_data = deque(self.splitIntoFolds(remain, num_folds, random_seed))
		folds = []
		for i in range(0, num_folds):
			curr = folded_data.popleft()
			info = self.getTestTraining(curr, folded_data)
			folds.append(info)
			folded_data.append(curr)
		#print len(folds
		self.trainAndCreateMetaDataSet(folds)
		self.trainMetaModel()
		xtrain, ytrain = self.splitIntoXY(remain)
		fold = (xtrain, None, ytrain, None)
		self.trainLevelOneModels(fold)
		curr_res = self.runModel(holdout)
		print "Holdout Results: " + str(curr_res)
		curr_res["ID"] = self.algorithm_id
		curr_res["Name"] = self.algorithm_name
		self.results = curr_res
		return curr_res
		
	def splitIntoFolds(self, data, k, seed):
		shuffled_data = shuffle(data, random_state=seed)
		#print shuffled_data
		folds = []
		num_in_folds = len(data) / k
		start = 0
		end = num_in_folds - 1
		for i in range(0,k):
			fold = shuffled_data.iloc[start:end]
			start = end
			end = end + num_in_folds - 1
			#print fold
			folds.append(self.splitIntoXY(fold))
			
		return folds
	
	def getTestTraining(self, curr, others):
		xtest = curr[0]
		ytest = curr[1]
		
		xtrainsets = []
		ytrainsets = []

		for curr in others:
			xtrainsets.append(pandas.DataFrame(curr[0]))
			ytrainsets.append(pandas.DataFrame(curr[1]))

		xtrain = pandas.concat(xtrainsets)
		ytrain = pandas.concat(ytrainsets)
		return xtrain, xtest, ytrain, ytest
	
	def crossValidateMetaModel(self, k):
		pass
		
	def getName(self):
		return self.algorithm_name
		
	def splitMetaIntoXY(self, data):
		self.resetKeys()
		#print data
		y = data[self.kb.Y]
		x = data[self.keys]
		try:
			x_cols = list(set(self.kb.X) - set(self.keys))
			features = data[x_cols]
		except:
			features = None
		return(x,y,features)
		
	def splitIntoAttributesOther(self, data):
		if data is not None:
			atr = list(set(self.kb.X) - set(self.other_predictions))
			x = data[atr]
			other = data[self.other_predictions]
			return(x,other)
		else:
			return (None, None)

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
		if self.other_predictions is not None:
			for name in self.other_predictions:
				self.keys.append(name)