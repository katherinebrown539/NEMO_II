from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase

##############################################################################################################
# ML-Controller class																					     #
# SKLearn interface for NEMO																		 		 #
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
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
		print stmt
		#self.data = read_sql_query(stmt, kb.db)
		stmt = "select " + kb.Y + " from DATA"
		print stmt
		#self.target = read_sql_query(stmt, kb.db)