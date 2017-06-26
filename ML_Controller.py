from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase
import NeuralNetworkController
import SciKit_Controller
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
	def __init__(self, kb):
		cols = ",".join(kb.X)
		stmt = "select " + cols + " from DATA;"
		#print stmt
		self.data = pandas.read_sql_query(stmt, kb.db)
		#print self.data
		print "data length = " + str(len(self.data))
		stmt = "select " + kb.Y + " from DATA"
		#print stmt
		self.target = pandas.read_sql_query(stmt, kb.db)
		#print self.target
		print "target length = " + str(len(self.target))
		self.kb = kb
		self.algorithm = NeuralNetworkController.NeuralNetworkController()
		
	def runAlgorithm(self):
		
		self.algorithm.createModel(self.data, self.target)
		results = self.algorithm.runModel()
		
		stmt = "insert into AlgorithmResults(algorithm_id, algorithm_name, accuracy, prec, recall, f1, confusion_matrix) values (%s,%s,%s,%s,%s,%s,%s)"
		self.kb.cursor.execute(stmt, results)
		self.kb.db.commit()
		
	def optimizeAlgorithm(self):
		self.algorithm.optimize()
		