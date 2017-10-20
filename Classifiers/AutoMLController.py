import autosklearn.classification
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

#THIS SHOULD ONLY BE USED WITH THE NELController!!
#NOT ALL METHODS REQUIRED FOR Nemo Explorer will be included in the first iteration
class AutoMLController:

	def __init__(self, kb, algorithm):
		#algorithm in ['']
		includes = []
		includes.append(algorithm)
		self.algorithm_name = "Auto " + algorithm + "_" + kb.name
		self.algorithm_id = ""
		random.seed()
		for i in range(1,10):
			self.algorithm_id = self.algorithm_id + str(random.randint(1,9))
		self.tree = None
		self.kb = kb
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1 = None
		self.cm = None
		self.mlp = None
		self.x = []
		self.y = []
		self.auto = None #autosklearn.classification.AutoSklearnClassifier(include_estimators = includes)

	def createModel(self):
		cols = ",".join(kb.X)
		stmt = "select " + cols + " from " + kb.name + ";"
		self.x = pandas.read_sql_query(stmt, kb.db)
		stmt = "select " + kb.Y + " from " + kb.name
		print stmt
		self.y = pandas.read_sql_query(stmt, kb.db)
		#self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y)
		self.auto = autosklearn.classification.AutoSklearnClassifier(include_estimators = includes, resampling_strategy='cv', resampling_strategy_arguments={'folds': 10})

		#self.tree.fit(self.X_train, self.y_train)



	def runModel(self, multi=False, x = None, y = None):
		#10 fold cv
		X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
		self.auto.fit(X_train.copy(), y_train.copy())

		self.auto.refit(X_train.copy(), y_train.copy())
		results = {}
		#GET RIGHT SCORES
		predictions = automl.predict(X_test)
		# accuracy
		results['Accuracy'] = sklearn.metrics.accuracy_score(y_test, predictions)
		# precision recall f1 support
		results['Precision'], results['Recall'], results['F1'], results['Support'] = sklearn.metrics.precision_recall_fscore_support(y_test, predictions)
		# roc
		results['ROC'] = sklearn.metrics.roc_curve(y_test, predictions)
		results['ROC_AUC'] = sklearn.metrics.roc_auc_score(y_test, predictions)
		# auc
		self.results = results
		print(results)
		#algorithm_id, algorithm_name, data_source, accuracy, prec, recall, f1
		kb.updateDatabaseWithResults(kb)
		return results

	def predict(self, x):
		return self.auto.predict(x)

	def fit(self,x,y):
		self.auto.fit(x,y)

	def isModelCreated(self):
		return self.auto is not None
		
	def createModelPreSplit(self, xtrain, xtest, ytrain, ytest, attributes=None):
		self.auto.fit(xtrain, ytrain)



################################ UNNEEDED FOR NELController #############################################
	def set_params(self, attr):
		pass

	def get_params(self):
		pass

	def optimize(self, metric, method):
		pass


	def createModelFromID(self, x, y, id):
		pass

	def copyModel(self,x,y,id):
		pass
