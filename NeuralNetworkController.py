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
	def __init__(self, x, y, layers=None):
		self.algorithm_name = "Neural Network"
		self.id = "ANN"
		random.seed()
		for i in range(1,10):
			self.id = str(random.randint(1,9))
		self.id = "( " + str(layers).strip('[]') + ")"	
		
		#initialize remaining instance variables
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1 = None
		self.cm = None
		self.layerslist = []
		self.x = x
		self.y = y
		
	def createModel(self):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x,self.y)
		scaler = StandardScaler()
		scaler.fit(self.X_train)
		self.X_train = scaler.transform(X_train)
		self.X_test = scaler.transform(X_test)
		
		if layers is None:
			random.seed()
			num_layers = random.randint(1,25)
			self.layerslist = []
			for i in range(0,num_layers):
				self.layerslist.append(random.randint(1,100))
		else:
			self.layerslist = layers
			
		self.mlp = MLPClassifier(hidden_layer_sizes=layers)
		mlp.fit(self.X_train, self.y_train)
		
	def runModel(self):
		predictions = mlp.predict(self.X_test)	
		
		print(confusion_matrix(self.y_test,predictions))
		print(classification_report(self.y_test,predictions))
		
		self.accuracy = accuracy_score(self.y_test,predictions)
		self.precision = precision_score(self.y_test,predictions)
		self.recall = recall_score(self.y_test, predictions)
		self.f1 = f1_score(self.y_test,predictions)
		self.cm = confusion_matrix(self.y_test,predictions)
		
		to_return =  (self.algorithm_id, self.algorithm_name, self.accuracy, self.precision, self.recall, self.f1, self.cm)
		return to_return
		