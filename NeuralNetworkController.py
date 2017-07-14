from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase
import random

class NeuralNetworkController:

	#Creates and processes a neural_network with defined architecture, or a random architecture
	#Preconditions:
	# * X - attributes as retrieved from the DATA table 
	# * Y - classes as retrieved from the the DATA table
	# * layers - architecture, may be none
	#Postconditions: returns performance from the neural network
	#NOTE: Code from kdnuggets
	def __init__(self, kb):
		self.algorithm_name = "Neural Network"
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

		self.x = []
		self.y = []
	
	def createModelFromID(self, x, y, id):
		self.algorithm_id = id
		stmt = "select * from ModelRepository where algorithm_id = " + self.algorithm_id
		# print stmt
		self.kb.executeQuery(stmt)
		row = self.kb.fetchOne()
		attributes = {}
		while row != None:
			key = row[2]
			val = row[3]
			print key + ": " + val
			if val == 'None' or val == 'NULL' or val is None:
				val = None
			else:
				if key in ['alpha', 'tol', 'momentum', 'validation_fraction', 'beta_1', 'beta_2', 'epsilon', 'learning_rate_init', 'power_t']:
					val = float(val)
				elif key in ['batch_size', 'max_iter', 'random_state']:
					if val != 'auto':
						val = int(val)
				elif key in ['shuffle', 'verbose', 'warm_start', 'nesterovs_momentum', 'early_stopping']:
					val = bool(val)
				elif key == 'hidden_layer_sizes':
					val = self.convertStringToTuple(val)
			attributes[key] = val
			row = self.kb.fetchOne()	

			
		self.createModel(x,y, attributes)
	
	def copyModel(self,x,y,id):
		temp = self.algorithm_id
		self.createModelFromID(x,y,id)
		self.algorithm_id = temp
		
	def createModel(self, x, y, attributes=None):
		# # print "X length " + str(len(x))
		# # print "Y length " + str(len(y))
		self.x = x
		self.y = y
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y)
		
		scaler = StandardScaler()
		scaler.fit(self.X_train)
		self.X_train = scaler.transform(self.X_train)
		self.X_test = scaler.transform(self.X_test)
		
		self.mlp = MLPClassifier()
		if attributes is None:
			attr = self.get_params()
			size = random.randint(1,10)
			layerslist = self.generateRandomArchitecture(size)
			attr['hidden_layer_sizes'] = tuple(layerslist)
			self.set_params(**attr)
		else:
			self.set_params(**attributes)	
		
		self.mlp.fit(self.X_train, self.y_train)
		
	
	def runModel(self):
		predictions = self.mlp.predict(self.X_test)	
		
		accuracy = accuracy_score(self.y_test,predictions)
		precision = precision_score(self.y_test,predictions, average='micro')
		recall = recall_score(self.y_test, predictions, average='micro')
		f1 = f1_score(self.y_test,predictions, average='micro')
		cm = confusion_matrix(self.y_test,predictions)
		
		self.results = {'ID': self.algorithm_id, 'Name': self.algorithm_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Confusion_Matrix': cm}
		
		to_return =  (self.algorithm_id, self.algorithm_name, accuracy, precision, recall, f1, cm)
		self.removeCurrentModel()
		return to_return
	
	
	def optimize(self, metric, method):
		if(method=='Coordinate Ascent'):
			return self.coordinateAscent(metric)
	
	def coordinateAscent(self, metric):
		bestModel = self
		while True:
			next = bestModel.optimizeNumberOfNodes(metric, bestModel)
			current_params = next.get_params()
			layers = current_params['hidden_layer_sizes']
			if bestModel.testForConvergence(layers): 
				#bestModel = self
				break
			else:
				bestModel = next 
				#update Model information...
				
		# print "Done optimizing this model"
		bestModel = bestModel.optimizeNumberOfLayers(metric, bestModel)
		
		
		return bestModel
		
	def optimizeNumberOfLayers(self, metric, best_model):
		best_attr = best_model.get_params()
		small_net_sz = len(best_attr['hidden_layer_sizes']) - 1
		large_net_sz = len(best_attr['hidden_layer_sizes']) + 1
		# attr = self.mlp.get_params()
		# size = random.randint(1,10)
		# layerslist = self.generateRandomArchitecture(size)
		# attr['hidden_layer_sizes'] = tuple(layerslist)
		
		small_net = NeuralNetworkController(self.kb)
		large_net = NeuralNetworkController(self.kb)
		small_arch = best_attr
		small_arch['hidden_layer_sizes'] = tuple(self.generateRandomArchitecture(small_net_sz))
		large_arch = best_attr
		large_arch['hidden_layer_sizes'] = tuple(self.generateRandomArchitecture(large_net_sz))
		
		small_net.createModel(self.x, self.y, small_arch)
		large_net.createModel(self.x, self.y, large_arch)
		small_net.runModel()
		large_net.runModel()
		
		small_net.optimizeNumberOfNodes(metric, small_net)
		large_net.optimizeNumberOfNodes(metric, large_net)
		
		#change metrics stuff
		#net.results.get(metric)
		#may want to change how comparisons get done here...
		if(large_net.results.get(metric) >= self.results.get(metric) and large_net.results.get(metric) >= small_net.results.get(metric)):
			return large_net
		if(small_net.results.get(metric) >= self.results.get(metric) and small_net.results.get(metric) >= large_net.results.get(metric)):
			return small_net
		else:
			return self
			
	def optimizeNumberOfNodes(self, metric, best_model):
		best_attr = best_model.get_params()
		current = best_attr['hidden_layer_sizes']
		curr_len = len(current)
		random.seed()
		percents = random.sample(xrange(1,100), curr_len)
		
		new_layers_inc = []
		for i in range(0, curr_len):
			curr = 1 + (percents[i]/100.0);
			new_layers_inc.append(int(1 + (curr * current[i])))
				
		new_layers_dec = []
		for i in range(0, curr_len):
			curr = 1 - (percents[i]/100.0);
			new_l = int(curr * current[i]);
			if new_l < 1: new_l = 1;
			new_layers_dec.append(new_l)
		
		
		increase_nn = NeuralNetworkController(self.kb)
		increase_arch = best_attr
		increase_arch['hidden_layer_sizes'] = tuple(new_layers_inc)
		increase_nn.createModel(self.x, self.y, increase_arch)
		increase_nn.runModel()
		
		decrease_nn = NeuralNetworkController(self.kb)
		decrease_arch = best_attr
		decrease_arch['hidden_layer_sizes'] = tuple(new_layers_dec)
		decrease_nn.createModel(self.x, self.y, decrease_arch)
		decrease_nn.runModel()
		
		
		#may want to change how comparisons get done here...
		if(increase_nn.results.get(metric) >= self.results.get(metric) and increase_nn.results.get(metric) >= decrease_nn.results.get(metric)):
			return increase_nn
		elif(decrease_nn.results.get(metric) >= self.results.get(metric) and decrease_nn.results.get(metric) >= increase_nn.results.get(metric)):
			return decrease_nn
		else:
			return self
			
	def testForConvergence(self, other): 
		me = self.get_params()
		return other == me['hidden_layer_sizes']
		
	def generateRandomArchitecture(self, num):
		to_return = []
		random.seed()
		for i in range(0, num):
			to_return.append(random.randint(1,20))
			
		return to_return
	
	def set_params(self, attr):
		self.mlp.set_parms(**attr)
		
	def get_params(self):
		return self.mlp.get_params()
	
	def convertStringToTuple(self, val):
		val = val.strip('( )')
		val = val.split(",")
		while val.count(''):
			val.remove('')
		val = map(int, val)
		val = tuple(val)
		return val 
		
		
	def removeModelFromRepository(self):
		stmt = "delete from ModelRepository where algorithm_id = " + self.algorithm_id
		self.kb.executeQuery(stmt)

	def updateDatabaseWithModel(self):
		arguments = self.get_params()
		#print arguments
		for key, value in arguments.iteritems():
			#print key + ": " + str(value)
			stmt = "insert into ModelRepository (algorithm_id, algorithm_name, arg_type, arg_val) values ( %s, %s, %s, %s)"
			args = (self.algorithm_id, self.algorithm_name, key, str(value))
			self.kb.executeQuery(stmt, args)
		
	def addCurrentModel(self):
		stmt = "insert into CurrentModel(algorithm_id) values (%s)"
		args = (self.algorithm_id,)
		self.kb.executeQuery(stmt, args)
		
	def removeCurrentModel(self):
		stmt = "delete from CurrentModel where algorithm_id = " + self.algorithm_id
		self.kb.executeQuery(stmt)