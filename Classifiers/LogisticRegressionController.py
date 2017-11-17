from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase
import random
import numpy

class LogisticRegression:

	def __init__(self, kb, c = 1.0):
		self.c = c
		self.algorithm_name = "Logistic Regression"
		self.algorithm_id = ""
		random.seed()
		for i in range(1,10):
			self.algorithm_id = self.algorithm_id + str(random.randint(1,9))

		self.kb = kb
		#initialize remaining instance variables
		self.X_train = []
		self.X_test = []
		self.y_train = []
		self.y_test = []
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1 = None
		self.cm = None
		self.mlp = None
		self.x = []
		self.y = []
		self.lr = LogisticRegression(penalty = "l1", C = c)

	def createModelFromID(self, x, y, id):
		pass

	def copyModel(self,x,y,id):
		pass

	def createModel(self, x, y, attributes=None):
		# # print "X length " + str(len(x))
		# # print "Y length " + str(len(y))
		self.x = x
		self.y = y
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y)
		self.lr = LogisticRegression(penalty = "l1", C = c)




		self.lr.fit(self.X_train, self.y_train)


	def createModelPreSplit(self, xtrain, xtest, ytrain, ytest, attributes=None):
		pass

	def runModel(self, multi=True, x = None, y = None):
		#self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x,self.y)
		self.lr.fit(self.X_train, self.y_train)

		# c, r = self.y.shape
		# labels = self.y.values.reshape(c,)
		# predictions = cross_val_predict(self.mlp, self.x, labels)
		# accuracy_all = cross_val_score(self.mlp, self.x, labels, cv=10)
		# accuracy = numpy.mean(accuracy_all)
		predictions = self.mlp.predict(self.X_test)
		accuracy = accuracy_score(self.y_test,predictions)
		precision = precision_score(self.y_test,predictions, average=av)
		recall = recall_score(self.y_test, predictions, average=av)
		f1 = f1_score(self.y_test,predictions, average=av)
		cm = confusion_matrix(self.y_test,predictions)

		self.results = {'ID': self.algorithm_id, 'Name': self.algorithm_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Confusion_Matrix': cm}

		to_return =  (self.algorithm_id, self.algorithm_name, accuracy, precision, recall, f1, cm)
		self.kb.removeCurrentModel(self)
		return to_return

	def predict(self, x):
		return self.lr.predict(x)

	def fit(self,x,y):
		self.lr.fit(x,y)


	def optimize(self, metric, method):
		pass

	def coordinateAscent(self, metric):
		pass


	def isModelCreated(self):
		return self.lr is not None

	def set_params(self, attributes):
		pass

	def get_params(self):
		pass

	def convertStringToTuple(self, val):
		pass
