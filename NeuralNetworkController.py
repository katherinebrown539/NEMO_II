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
		
		
		self.x = []
		self.y = []
		
	def createModel(self, x, y,size = None, layers=None):
		print "X length " + str(len(x))
		print "Y length " + str(len(y))
		self.x = x
		self.y = y
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,y)
		
		scaler = StandardScaler()
		scaler.fit(self.X_train)
		self.X_train = scaler.transform(self.X_train)
		self.X_test = scaler.transform(self.X_test)
		
		if size is None:
			if layers is None:
				random.seed()
				#creates a hidden architecture of up to 10 layers where each layer can have up to 10 nodes
				self.layerslist = random.sample(xrange(1,20), random.randint(1,10))
			elif layers is not None:
				self.layerslist = random.sample(xrange(1,20), size)
			else: self.layerslist = layers
		
		print str(self.layerslist)
		
		self.algorithm_id = self.algorithm_id_abbr + self.id +  "( " + str(self.layerslist).strip('[]') + ")"	
		self.mlp = MLPClassifier(hidden_layer_sizes=self.layerslist)
		self.mlp.fit(self.X_train, self.y_train)
	
	
	def runModel(self):
		predictions = self.mlp.predict(self.X_test)	
		
		print(confusion_matrix(self.y_test,predictions))
		print(classification_report(self.y_test,predictions))
		
		self.accuracy = accuracy_score(self.y_test,predictions)
		self.precision = precision_score(self.y_test,predictions)
		self.recall = recall_score(self.y_test, predictions)
		self.f1 = f1_score(self.y_test,predictions)
		self.cm = confusion_matrix(self.y_test,predictions)
		
		self.results =  (self.algorithm_id, self.algorithm_name, self.accuracy, self.precision, self.recall, self.f1, self.cm)
		return self.results
	
	def optimize(self):
		self.optimizeNumberOfNodes()
		small_net_sz = self.len(layerslist) - 1
		large_net_sz = self.len(layerslist) + 1
		
		small_net = NeuralNetworkController()
		large_net = NeuralNetworkController()
		small_net.createModel()
		large_net.createModel()
		small_net.runModel()
		large_net.runModel()
		small_net.optimizeModel()
		large_net.optimizeModel()
		
		if(large_net.accuracy >= self.accuracy and large_net.accuracy >= small_net.accuracy):
			print "Larger Model wins"
			print large_net
			return large_net
		if(small_net.accuracy >= self.accuracy and small_net.accuracy >= large_net.accuracy):
			print "Smaller Model wins"
			print small_net
			return small_net
		else:
			print "Same model wins"
			print self
			return self
	
	def optimizeNumberOfNodes(self):
		#pick random percentages for each layer, this varies shape
		random.seed()
		percents = random.sample(xrange(1,100), len(self.layerslist))
		print "Percents: " + str(percents)
		print "Current architecture: " + str(self.layerslist)
		
		#increase hidden layers by those percentages
		new_layers_inc = []
		for i in range(0, len(self.layerslist)):
			curr = 1 + (percents[i]/100.0);
			new_layers_inc.append(int(1 + (curr * self.layerslist[i])))
				
		print "Increased architecture: " + str(new_layers_inc)
		#decrease hidden layers by those percentages
		new_layers_dec = []
		for i in range(0, len(self.layerslist)):
			curr = 1 - (percents[i]/100.0);
			new_l = int(curr * self.layerslist[i]);
			if new_l < 1: new_l = 1;
			new_layers_dec.append(new_l)
		
		print "Decreased architecture: " + str(new_layers_dec)
		
		#create new models, and compare
		increase_nn = NeuralNetworkController()
		increase_nn.createModel(self.x, self.y, new_layers_inc)
		increase_nn.runModel()
		
		decrease_nn = NeuralNetworkController()
		decrease_nn.createModel(self.x, self.y, new_layers_dec)
		decrease_nn.runModel()
		
		print "Accuracy of current model: " + str(self.accuracy)
		print "Accuracy of increased model: " + str(increase_nn.accuracy)
		print "Accuracy of decreased model: " + str(decrease_nn.accuracy)
		
		if(increase_nn.accuracy >= self.accuracy and increase_nn.accuracy >= decrease_nn.accuracy):
			print "Increased Model wins"
			print increase_nn
			return increase_nn
		if(decrease_nn.accuracy >= self.accuracy and decrease_nn.accuracy >= increase_nn.accuracy):
			print "Decreased Model wins"
			print decrease_nn
			return decrease_nn
		else:
			print "Same model wins"
			print self
			return self
			
	def testForConvergence(self, other): 
		return other == self.layerslist
		
	#def optimizeLearningRate(self):
	
	