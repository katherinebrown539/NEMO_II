from sklearn.svm import SVC
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

class SVMController:

	def __init__(self, kb):
		self.algorithm_name = "SVM"
		random.seed()
		for i in range(1,10):
			self.algorithm_id = self.algorithm_id + str(random.randint(1,9))
		self.svm = None
		self.kb = kb

	def createModel(self, x, y, attributes=None):
		self.x = x
		self.y = y
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y)

		if attributes is not None:
			self.svm = SVC()
			self.set_params(attributes)
		else:
			self.svm = SVC(probability=True)

		self.svm.fit(self.X_train, self.y_train)


	def createModelPreSplit(self, xtrain, xtest, ytrain, ytest, attributes=None):
		self.X_train = xtrain
		self.X_test = xtest
		self.y_train = ytrain
		self.y_test = ytest

		if attributes is not None:
			self.svm = SVC()
			self.set_params(attributes)
		else:
			self.svm = SVC(probability=True)

		self.svm.fit(self.X_train, self.y_train)

	def createModelFromID(self, x, y, id):
		#run query
		stmt = "select * from ModelRepository where algorithm_id = " + id
		self.kb.executeQuery(stmt)
		row = self.kb.fetchOne()
		attributes = {}
		while row != None:
			#print row
			key = row[2]
			val = row[3]
			#print key + ": " + val
			if val == 'None' or val == 'NULL':
				val = None
			if val is not None:
				if key == "DATA_SOURCE":
					row = self.kb.fetchOne()
					continue
				elif key in ['C', 'gamma', 'coef0', 'tol', 'cache_size']:
					if key == 'gamma' and val == 'auto':
						val = val
					else:
						val = float(val)
				elif key in ['degree', 'max_iter', 'random_state']:
					val = int(val)
				elif key in ['probability', 'shrinking', 'verbose']:
					val = bool(val)
			#print type(val)
			attributes[key] = val
			row = self.kb.fetchOne()
		self.createModel(x,y,attributes)

	def copyModel(self,x,y,id):
		self.algorithm_id = id
		self.createModelFromID(x,y,id)

	def runModel(self, multi=False, x = None, y = None):
		#self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x,self.y)
		self.svm.fit(self.X_train, self.y_train)
		if x is not None:
			self.X_test = x
			self.y_test = y
		av = 'micro'
		# if not multi:
			# av = 'binary'
		# else:
			# av = 'micro'
		# c, r = self.y.shape
		# labels = self.y.values.reshape(c,)
		# predictions = cross_val_predict(self.svm, self.x, labels)
		# accuracy_all = cross_val_score(self.svm, self.x, labels, cv=10)
		# accuracy = numpy.mean(accuracy_all)
		predictions = self.svm.predict(self.X_test)
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
		return self.svm.predict(x)

	def fit(self,x,y):
		self.svm.fit(x,y)

	def isModelCreated(self):
		return self.svm is not None

	def set_params(self, attr):
		self.svm.set_params(**attr)

	def get_params(self):
		return self.svm.get_params()

	def optimize(self, metric, method):
		if method == 'Coordinate Ascent':
			return self.coordinateAscent(metric)

	def coordinateAscent(self, metric):
		best_model = self
		bst = 0
		curr = best_model.results.get(metric)
		curr_mdl = best_model
		while curr > bst:
			bst = curr
			best_model = curr_mdl
			self.kb.updateDatabaseWithModel(best_model)
			curr_mdl = self.optimizeC(metric, self)
			curr = curr_mdl.results.get(metric)
		current_model = best_model
		while curr > bst:
			bst = curr
			best_model = current_model
			current_model = self.optimizeKernel(metric, best_model)
			#print "current_model @ coordinateAscent: " + str(current_model)
			curr = current_model.results.get(metric)
		self.kb.updateDatabaseWithModel(best_model)
		return best_model

	def optimizeC(self, metric, best_model):
		attributes = best_model.get_params()
		percent = random.randint(1,100)
		percent = percent/100.0

		inc_C = attributes['C'] * (1.0+percent)
		dec_C = attributes['C'] * (percent)

		inc_attr = attributes
		inc_attr['C'] = inc_C
		dec_attr = attributes
		dec_attr['C'] = dec_C

		inc_model = SVMController(self.kb)
		dec_model = SVMController(self.kb)

		inc_model.createModel(self.x, self.y, inc_attr)
		dec_model.createModel(self.x, self.y, dec_attr)

		inc_model.runModel()
		dec_model.runModel()

		if inc_model.results[metric] >= best_model.results[metric] and inc_model.results[metric] > dec_model.results[metric]:
			return inc_model
		elif dec_model.results[metric] >= best_model.results[metric] and dec_model.results[metric] > inc_model.results[metric]:
			return dec_model
		else:
			return best_model

	def optimizeKernel(self, metric, best_model):
		kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
		#get best model kernel
		attr = best_model.get_params()
		kernels.remove(attr['kernel'])
		best_metric = best_model.results[metric]
		while len(kernels) > 0:
			new_mdl_attr = attr
			new_mdl_attr['kernel'] = kernels.pop()
			new_mdl = SVMController(self.kb)
			new_mdl.createModel(self.x, self.y, new_mdl_attr)
			new_mdl.runModel()
			if new_mdl.results.get(metric) >= best_metric:
				best_model = new_mdl
				best_metric = new_mdl.results.get(metric)
		return best_model
