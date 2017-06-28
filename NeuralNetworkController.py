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
	def __init__(self):
		self.algorithm_name = "Neural Network"
		self.algorithm_id_abbr = "ANN"
		self.id = ""
		random.seed()
		for i in range(1,10):
			self.id = self.id + str(random.randint(1,9))
		
		
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
		self.layerslist = []
		
		self.x = []
		self.y = []
		
	def createModel(self, x, y, layers=None, size = None,):
		# print "X length " + str(len(x))
		# print "Y length " + str(len(y))
		self.x = x
		self.y = y
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y)
		
		scaler = StandardScaler()
		scaler.fit(self.X_train)
		self.X_train = scaler.transform(self.X_train)
		self.X_test = scaler.transform(self.X_test)
		
		if size is None:
			size = random.randint(1,10)
			
		if layers is not None: #predefined architecture
			self.layerslist = layers
		else: self.layerslist = self.generateRandomArchitecture(size)
		
		#print str(self.layerslist)
		
		self.algorithm_id = self.algorithm_id_abbr + self.id +  "( " + str(self.layerslist).strip('[]') + ")"	
		self.mlp = MLPClassifier(hidden_layer_sizes=self.layerslist)
		self.mlp.fit(self.X_train, self.y_train)
	
	
	def runModel(self):
		#print "Architecture of model: " + str(self.layerslist)
		predictions = self.mlp.predict(self.X_test)	
		
		#print(confusion_matrix(self.y_test,predictions))
		#print(classification_report(self.y_test,predictions))
		
		accuracy = accuracy_score(self.y_test,predictions)
		precision = precision_score(self.y_test,predictions)
		recall = recall_score(self.y_test, predictions)
		f1 = f1_score(self.y_test,predictions)
		cm = confusion_matrix(self.y_test,predictions)
		
		self.results = {'ID': self.algorithm_id, 'Name': self.algorithm_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Confusion_Matrix': cm}
		
		to_return =  (self.algorithm_id, self.algorithm_name, accuracy, precision, recall, f1, cm)
		return to_return
	
	
	def optimize(self, metric, method):
		if(method=='Coordinate Ascent'):
			return self.coordinateAscent(metric)
	
	def coordinateAscent(self, metric):
		bestModel = self
		while True:
			next = bestModel.optimizeNumberOfNodes(metric)
			if bestModel.testForConvergence(next.layerslist): 
				bestModel = self
				break
			else:
				bestModel = next
				
		print "Done optimizing this model"
		bestModel.optimizeNumberOfLayers(metric)
		
		
		return bestModel
		
	def optimizeNumberOfLayers(self, metric):
		small_net_sz = len(self.layerslist) - 1
		large_net_sz = len(self.layerslist) + 1
		
		small_net = NeuralNetworkController()
		large_net = NeuralNetworkController()
		small_net.createModel(self.x, self.y, None, small_net_sz)
		large_net.createModel(self.x, self.y, None, large_net_sz)
		small_net.runModel()
		large_net.runModel()
		small_net.optimizeNumberOfNodes(metric)
		large_net.optimizeNumberOfNodes(metric)
		
		#change metrics stuff
		#net.results.get(metric)
		#may want to change how comparisons get done here...
		if(large_net.results.get(metric) >= self.results.get(metric) and large_net.results.get(metric) >= small_net.results.get(metric)):
			print "Larger Layers Model wins"
			print large_net
			return large_net
		if(small_net.results.get(metric) >= self.results.get(metric) and small_net.results.get(metric) >= large_net.results.get(metric)):
			print "Smaller Layers Model wins"
			print small_net
			return small_net
		else:
			print "Same Layers Model wins"
			print self
			return self
			
	def optimizeNumberOfNodes(self, metric):
	
		random.seed()
		percents = random.sample(xrange(1,100), len(self.layerslist))
		print "Percents: " + str(percents)
		print "Current architecture: " + str(self.layerslist)
		
		#increase hidden layers by those percentages
		new_layers_inc = []
		for i in range(0, len(self.layerslist)):
			curr = 1 + (percents[i]/100.0);
			new_layers_inc.append(int(1 + (curr * self.layerslist[i])))
				
		print "Increased nodes architecture: " + str(new_layers_inc)
		#decrease hidden layers by those percentages
		new_layers_dec = []
		for i in range(0, len(self.layerslist)):
			curr = 1 - (percents[i]/100.0);
			new_l = int(curr * self.layerslist[i]);
			if new_l < 1: new_l = 1;
			new_layers_dec.append(new_l)
		
		print "Decreased nodes architecture: " + str(new_layers_dec)
		
		#create new models, and compare
		increase_nn = NeuralNetworkController()
		increase_nn.createModel(self.x, self.y, new_layers_inc)
		increase_nn.runModel()
		
		decrease_nn = NeuralNetworkController()
		decrease_nn.createModel(self.x, self.y, new_layers_dec)
		decrease_nn.runModel()
		
		#change metrics stuffs
		#.results.get(metric)
		print "Accuracy of current model: " + str(self.results.get(metric))
		print "Accuracy of increased nodes model: " + str(increase_nn.results.get(metric))
		print "Accuracy of decreased nodes model: " + str(decrease_nn.results.get(metric))
		
		#may want to change how comparisons get done here...
		if(increase_nn.results.get(metric) >= self.results.get(metric) and increase_nn.results.get(metric) >= decrease_nn.results.get(metric)):
			print "Increased nodes Model wins"
			return increase_nn
		elif(decrease_nn.results.get(metric) >= self.results.get(metric) and decrease_nn.results.get(metric) >= increase_nn.results.get(metric)):
			print "Decreased nodes Model wins"
			return decrease_nn
		else:
			print "Same nodes model wins"
			return self
			
	def testForConvergence(self, other): 
		return other == self.layerslist
		
	def generateRandomArchitecture(self, num):
		to_return = []
		random.seed()
		for i in range(0, num):
			to_return.append(random.randint(1,20))
			
		return to_return
	
	