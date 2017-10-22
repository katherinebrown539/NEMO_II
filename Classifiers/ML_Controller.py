from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase
from Classifiers import NeuralNetworkController, DecisionTreeController, RandomForestController, SVMController
import random
##############################################################################################################
# ML-Controller class																					     #
# SKLearn interface for NEMO																		 		 #
##############################################################################################################
# ***** INSTANCE VARIABLES*****																				 #
# data		-		attributes as retrieved from the DATA table 											 #
# target	-		classes as retrieved from the the DATA table											 #
# kb		-		instance of knowledge base object														 #
##############################################################################################################
class ML_Controller:
	#Constructor
	#imports data from knowledge base
	#Preconditions:
	# * A knowledge base has been set up and read data
	#Postconditions:
	# * Data will be imported from the
	def __init__(self, kb, algorithm_type):
		print (kb.X)
		cols = ",".join(kb.X)
		stmt = "select " + cols + " from " + kb.name + ";"
		print (stmt)
		self.data = pandas.read_sql_query(stmt, kb.db)
		#print self.data
		#print "data length = " + str(len(self.data))
		stmt = "select " + kb.Y + " from " + kb.name
		print (stmt)
		self.target = pandas.read_sql_query(stmt, kb.db)
		#print self.target
		#print "target length = " + str(len(self.target))
		self.kb = kb
		self.isCurrentlyOptimizing = False	#True when model is in optimization queue, false otherwise
		self.algorithm = None
		self.name = self.kb.name + "_" + algorithm_type
		print (algorithm_type)
		if algorithm_type == "Neural Network":
			self.algorithm = NeuralNetworkController.NeuralNetworkController(self.kb)
		elif algorithm_type == "Decision Tree":
			self.algorithm = DecisionTreeController.DecisionTreeController(self.kb)
		elif algorithm_type == 'SVM':
			self.algorithm = SVMController.SVMController(self.kb)
		elif algorithm_type == "Random Forest":
			self.algorithm = RandomForestController.RandomForestController(self.kb)
		else:
			self.algorithm = DecisionTreeController.DecisionTreeController(self.kb)

	def changeKB(self, new_kb):
		self.kb = new_kb
		if self.algorithm is not None:
			self.algorithm.kb = new_kb
		cols = ",".join(self.kb.X)
		stmt = "select " + cols + " from " + self.kb.name + ";"
		#print stmt
		self.data = pandas.read_sql_query(stmt, self.kb.db)
		#print self.data
		#print "data length = " + str(len(self.data))
		stmt = "select " + self.kb.Y + " from " + self.kb.name
		#print stmt
		self.target = pandas.read_sql_query(stmt, self.kb.db)

	def get_params(self):
		return self.algorithm.get_params()

	def createModel(self, id=None):
		if id is None:
			self.algorithm.createModel(self.data, self.target)
		else:
			self.algorithm.createModelFromID(self.data, self.target, id)

	def createModelPreSplit(self, xtrain, xtest, ytrain, ytest, attributes=None):
		if attributes is None and self.algorithm.isModelCreated():
			attributes = self.get_params()
		self.algorithm.createModelPreSplit(xtrain, xtest, ytrain, ytest, attributes)

	def copyModel(self, id):
		self.algorithm.copyModel(self.data, self.target, id)

	def fit(self, x, y):

		self.algorithm.fit(x,y)

	def predict(self, x):
		return self.algorithm.predict(x)

	def runAlgorithm(self, x = None, y = None):
		results = self.algorithm.runModel(self.kb.multi, x, y)
		#self.kb.updateDatabaseWithResults(self.algorithm)
		return results

	def updateDatabaseWithResults(self):
		self.kb.updateDatabaseWithResults(self.algorithm)

	def getName(self):
		return self.algorithm.algorithm_name

	def getID(self):
		return self.algorithm.algorithm_id

	def optimizeAlgorithm(self):
		curr_id = self.algorithm.algorithm_id
		self.algorithm = self.algorithm.optimize('Accuracy', 'Coordinate Ascent')
		self.algorithm.algorithm_id = curr_id
		self.algorithm.results['ID'] = curr_id
		self.kb.updateDatabaseWithResults(self.algorithm)
		#self.kb.removeModelFromRepository(self.algorithm)
		self.kb.updateDatabaseWithModel(self.algorithm)
