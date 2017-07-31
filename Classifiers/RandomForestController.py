from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase
import random
import numpy

class RandomForestController:

	def __init__(self, kb):	
		self.algorithm_name = "Random Forest"
		self.algorithm_id = ""
		random.seed()
		for i in range(1,10):
			self.algorithm_id = self.algorithm_id + str(random.randint(1,9))
		self.forest = None
		self.kb = kb
		
	def createModel(self, x, y, attributes=None):
		pass
	
	def createModelPreSplit(self, xtrain, xtest, ytrain, ytest, attributes=None):
		pass
	
	def createModelFromID(self, x, y, id):
		pass
	
	def copyModel(self,x,y,id):
		pass
	
	def runModel(self, multi=False, x = None, y = None,):
		pass
		
	def coordinateAscent(self, metric):
		pass
	
	def predict(self, x):
		return self.forest.predict(x)
	
	def fit(self,x,y):
		self.forest.fit(x,y)
	
	def set_params(self, attr):
		self.forest.set_params(**attr)
		
	def get_params(self):
		return self.forest.get_params()
		
	def optimize(self, metric, method):
		if method == 'Coordinate Ascent':
			return self.coordinateAscent(metric)

	def isModelCreated(self):
		return self.forest is not None		