from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pandas
from pandas import DataFrame
import pandas.io.sql as psql
import KnowledgeBase

#Creates and processes a neural_network with defined architecture, or a random architecture
#Preconditions:
# * X - attributes as retrieved from the DATA table 
# * Y - classes as retrieved from the the DATA table
# * layers - architecture, may be none
#Postconditions: returns performance from the neural network
#NOTE: Code from kdnuggets
def NeuralNetwork(x, y, layers=None):
	X_train, X_test, y_train, y_test = train_test_split(x,y)
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	
	mlp = MLPClassifier(hidden_layer_sizes=layers)
	mlp.fit(X_train,y_train)
	predictions = mlp.predict(X_test)
	
	
	print(confusion_matrix(y_test,predictions))
	print(classification_report(y_test,predictions))
	
	accuracy = metrics.accuracy_score(y_test,predictions)
	precision = metrics.precision_score(y_test,predictions)
	recall = metrics.recall_score(y_test, predictions)
	f1 = metrics.f1_score(y_test,predictions)
	cm = confusion_matrix(y_test,predictions)
	return accuracy,precision,recall,f1,cm