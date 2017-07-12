from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase
import NeuralNetworkController
import DecisionTreeController
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
		cols = ",".join(kb.X)
		stmt = "select " + cols + " from DATA;"
		#print stmt
		self.data = pandas.read_sql_query(stmt, kb.db)
		#print self.data
		#print "data length = " + str(len(self.data))
		stmt = "select " + kb.Y + " from DATA"
		#print stmt
		self.target = pandas.read_sql_query(stmt, kb.db)
		#print self.target
		#print "target length = " + str(len(self.target))
		self.kb = kb
		self.isCurrentlyOptimizing = False	#True when model is in optimization queue, false otherwise
		self.algorithm = None
		print algorithm_type
		if algorithm_type == "Neural Network":
			self.algorithm = NeuralNetworkController.NeuralNetworkController(self.kb)
		if algorithm_type == "Decision Tree":
			self.algorithm = DecisionTreeController.DecisionTreeController(self.kb)
		
	
	def createModel(self, id=None):
		if id is None:
			self.algorithm.createModel(self.data, self.target)
		else:
			self.algorithm.createModelFromID(self.data, self.target, id)
	
	def copyModel(self, id):
		self.algorithm.copyModel(self.data, self.target, id)
		
	def runAlgorithm(self):
		results = self.algorithm.runModel()
		#self.kb.updateDatabaseWithResults(self.algorithm)
		
	def updateDatabaseWithResults(self):
		self.kb.updateDatabaseWithResults(self.algorithm)
		
	def getID(self):
		return self.algorithm.algorithm_id
		
	def optimizeAlgorithm(self):
		curr_id = self.algorithm.algorithm_id
		self.algorithm = self.algorithm.optimize('Accuracy', 'Coordinate Ascent')
		self.algorithm.algorithm_id = curr_id
		self.algorithm.results['ID'] = curr_id
		self.kb.updateDatabaseWithResults(self.algorithm)
		self.algorithm.removeModelFromRepository()
		self.algorithm.updateDatabaseWithModel()