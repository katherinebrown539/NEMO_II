from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller
from collections import deque
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas, MySQLdb, threading, sys, os, time, random

class AutoKnowledgeIntegrator:
    def __init__(self, kb, level1_classifiers, stacking_classifier=None, use_features=False):
		self.kb = kb
		self.level1_classifiers = level1_classifiers
		if stacking_classifier is None or stacking_classifier == "Logistic Regression":
			self.algorithm_name = "KI_LogisticRegression"
			self.stacking_classifier = LogisticRegression()
		elif stacking_classifier == "Decision Tree":
			self.algorithm_name = "KI_DecisionTree"
			self.stacking_classifier = DecisionTreeClassifier()
		elif stacking_classifier == "SVM":
			self.algorithm_name = "KI_SVM"
			self.stacking_classifier = SVC()
		#self.keys.append(self.kb.Y)
		self.algorithm_id = "KI"+str(random.randint(1,101))
		self.use_features = use_features
		self.data = self.kb.getData()
		print("Data")
		print(self.data)
