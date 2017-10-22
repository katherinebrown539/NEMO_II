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
		self.x = x
		self.y = y
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y)

		if attributes is not None:
			self.forest = RandomForestClassifier(random_state=None)
			self.set_params(attributes)
		else:
			self.forest = RandomForestClassifier(random_state=0)

		self.forest.fit(self.X_train, self.y_train)

	def createModelPreSplit(self, xtrain, xtest, ytrain, ytest, attributes=None):
		self.X_train = xtrain
		self.X_test = xtest
		self.y_train = ytrain
		self.y_test = ytest

		if attributes is not None:
			self.forest = RandomForestClassifier(random_state=None)
			self.set_params(attributes)
		else:
			self.forest = RandomForestClassifier(random_state=0)

		self.forest.fit(self.X_train, self.y_train)


	def createModelFromID(self, x, y, id):
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
				if key == 'max_features':
					lst = list(val)
					if lst.count('.') > 0:
						val = float(val)
					elif val in ['auto', 'sqrt', 'log2']:
						val = val
					else:
						val = int(val)
				if key in ['max_depth', 'max_leaf_nodes', 'random_state', 'n_estimators', 'max_leaf_nodes', 'n_jobs', 'random_state', 'verbose']:
					val = int(val)
				elif key in ['min_weight_fraction_leaf', 'min_impurity_split']:
					val = float(val)
				elif key in ['min_samples_split', 'min_samples_leaf']:
					lst = list(val)
					if lst.count('.') > 0:
						val = float(val)
					else:
						val = int(val)
				elif key in ['boostrap', 'oob_score', 'warm_start']:
					val = (val == "True")
				elif key in ['class_weight']:
					lst = list(val)
					if lst.count('{') == 1:
						val = dict(val)
					elif lst.count('{') > 1:
						val = list(val)
						for i in range(0,len(val)):
							val[i] = dict(val[i])
			#print type(val)
			attributes[key] = val
			row = self.kb.fetchOne()
		self.createModel(x,y,attributes)


	def copyModel(self,x,y,id):
		self.algorithm_id = id
		self.createModelFromID(x,y,id)

	def runModel(self, multi=False, x = None, y = None,):
		#self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x,self.y)
		self.forest.fit(self.X_train, self.y_train)

		if x is not None:
			self.X_test = x
			self.y_test = y
		av = ''
		if not multi:
			av = 'binary'
		else:
			av = 'micro'

		#print "Number of test instances: " + str(len(self.X_test.index))


		predictions = self.forest.predict(self.X_test)
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

	def coordinateAscent(self, metric):
		best_model = self
		bst = 0.0
		curr = best_model.results.get(metric)
		current_model = best_model
		while curr > bst:
			bst = curr
			best_model = current_model
			current_model = self.optimizeCriterion(metric, self)
			#print "current_model @ coordinateAscent: " + str(current_model)
			curr = current_model.results.get(metric)
		self.kb.updateDatabaseWithModel(best_model)
		curr = best_model.results.get(metric)
		current_model = best_model
		while curr > bst:
			bst = curr
			best_model = curr
			current_model = self.optimizeMaxFeatures(metric, self)
			curr = current_model.results.get(metric)
		self.kb.updateDatabaseWithModel(best_model)
		curr = best_model.results.get(metric)
		current_model = best_model
		while curr > bst:
			bst = curr
			best_model = curr
			current_model = self.optimizeNumEstimators(metric, self)
			curr = current_model.results.get(metric)
		self.kb.updateDatabaseWithModel(best_model)

		return best_model

	def optimizeNumEstimators(self, metric, best_model):
		current_attr = best_model.get_params()
		current_num_estimators = current_attr.get('n_estimators')
		percent = random.uniform(0,1)
		num_inc = int((1 + percent) * current_num_estimators)
		num_dec = int(percent * current_num_estimators)

		inc_forest = RandomForestController(self.kb)
		inc_attr = current_attr
		inc_attr['n_estimators'] = num_inc
		inc_forest.createModel(self.x,self.y, inc_attr)
		dec_forest = RandomForestController(self.kb)
		dec_attr = current_attr
		dec_attr['n_estimators'] = num_dec
		dec_forest.createModel(self.x,self.y, dec_attr)

		if(inc_forest.results.get(metric) >= best_model.results.get(metric)) and (inc_forest.results.get(metric) >= dec_forest.results.get(metric)):
			return inc_forest
		if(dec_forest.results.get(metric) >= best_model.results.get(metric)) and (dec_forest.results.get(metric) >= inc_forest.results.get(metric)):
			return dec_forest
		else:
			return best_model


	def optimizeMaxFeatures(self, metric, best_model):
		best_model_attributes = best_model.get_params()
		#sqrt
		sqrt_forest = RandomForestController(self.kb)
		sqrt_attr = best_model_attributes
		sqrt_attr['max_features'] = 'sqrt'
		sqrt_forest.createModel(self.x, self.y, sqrt_attr)
		sqrt_forest.runModel(self.kb.multi)

		#log2
		log_forest = RandomForestController(self.kb)
		log_attr = best_model_attributes
		log_attr['max_features'] = 'log2'
		log_forest.createModel(self.x, self.y, log_attr)
		log_forest.runModel(self.kb.multi)

		#test between sqrt and log2
		if(sqrt_forest.results.get(metric) >= best_model.results.get(metric) and sqrt_forest.results.get(metric) >= log_forest.results.get(metric)):
			return sqrt_forest
		if(log_forest.results.get(metric) >= best_model.results.get(metric) and log_forest.results.get(metric) >= sqrt_forest.results.get(metric)):
			return log_forest
		else:
			return best_model
		#random percent
		best_metric = 0
		best_percent_model = None
		for i in range(1,100):
			curr = i/100.0
			percent_attr = best_model_attributes.get_params()
			percent_attr['max_features'] = curr
			percent_forest = RandomForestController(self.kb)
			percent_forest.createModel(self.x, self.y, percent_attr)
			percent_forest.runModel(self.kb.multi)
			if percent_forest.results.get(metric) >= best_metric:
				best_percent_model = percent_forest

		if best_percent_model.results.get(metric) >= best_model.results.get(metric):
			return best_percent_model
		else:
			return best_model

	def optimizeCriterion(self, metric, best_model):
		attributes = best_model.get_params()
		attributes['criterion'] = "entropy" if attributes['criterion'] == 'gini' else 'gini'

		criterion_forest = RandomForestController(self.kb)
		# print "criterion_forest: " + str(criterion_forest)
		# print "best_model: " +  str(best_model)
		criterion_forest.createModel(self.x, self.y, attributes)
		criterion_forest.runModel(self.kb.multi)
		if(criterion_forest.results.get(metric) >= best_model.results.get(metric)):
			# print "criterion_forest: " + str(criterion_forest)
			return criterion_forest
		else:
			# print "best_model: " +  str(best_model)
			return best_model
