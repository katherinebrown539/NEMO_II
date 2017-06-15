from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase
import SciKit_Controller

##############################################################################################################
# ML-Controller class																					     #
# SKLearn interface for NEMO																		 		 #
##############################################################################################################
# ***** INSTANCE VARIABLES*****																				 #
# data		-		attributes as retrieved from the DATA table 											 #
# target	-		classes as retrieved from the the DATA table											 #
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
		stmt = "select " + kb.Y + " from DATA"
		#print stmt
		self.target = pandas.read_sql_query(stmt, kb.db)
		#print self.target
	
	def runAlgorithm(self):
		SciKit_Controller.NeuralNetwork(self.data, self.target, (30,30,30))
		