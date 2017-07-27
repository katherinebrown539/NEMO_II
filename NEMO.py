#!/usr/bin/env python 
from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller, KnowledgeIntegrator
from collections import deque
from sklearn.model_selection import train_test_split
import pandas
import numpy
from sklearn.utils import shuffle
import MySQLdb
import threading
import sys
import os
import time
import json
#test comment for git
#one stop event, pass in the queue and number of seconds to spend optimizing
def optimizeAlgorithmWorker(ml, stp):
	while not stp.is_set():
		ml.optimizeAlgorithm()
	
def optimizeWorker(queue, stp, secs):
	while not stp.is_set():
		task = queue.popleft()
		#print "Optimizing " + task.getID()
		opt_stp = threading.Event()
		thrd = threading.Thread(target=optimizeAlgorithmWorker, args=(task, opt_stp))
		thrd.start()
		time.sleep(secs)
		opt_stp.set()
		thrd.join()
		queue.append(task)
		
	
class NEMO:
	def __init__(self, filename):
		self.kb = KnowledgeBase.KnowledgeBase(filename)
		self.ml = [] #list of machine learners
		self.queue = deque()
		self.optimization_thread = None
		self.stop_event = None
		self.checkForCurrentModels()
		self.checkForOptimizingModels()
		self.secs = 45
		with open(filename) as fd:
			json_data = json.load(fd)
		info = json_data['KNOWLEDGE_INTEGRATOR']
		self.stacking_classifier = info["STACKER"]
		self.other_predictions = info["OTHER_PREDICTIONS"] if info['OTHER_PREDICTIONS'] != "None" else None
		
	def findAlgorithmBasedOnID(self, id):
		for model in self.ml:
			if id == model.getID():
				return model
	
	def verifyID(self, id):
		stmt = "select algorithm_id from ModelRepository"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		ids = self.kb.fetchAll()
		
		return (id,) in ids
	
	def getAlgorithmType(self, id):
		#assumes id has already been verified
		stmt = "select algorithm_name from ModelRepository where algorithm_id = " + id
		self.kb.executeQuery(stmt)
		types = self.kb.fetchOne()
		return types[0]
		
	#same model, different id
	def createModelBasedONID(self):
		#self.printModelInformation()
		id = raw_input("Enter ID Here --> ")
		if self.verifyID(id):
			type = self.getAlgorithmType(id)
			new_ml = ML_Controller.ML_Controller(self.kb, type)
			new_ml.createModel(id)
			self.kb.updateDatabaseWithModel(new_ml.algorithm)	
			self.kb.addCurrentModel(new_ml.algorithm)
			new_ml.runAlgorithm()
			new_ml.updateDatabaseWithResults()
			self.ml.append(new_ml)
		else:
			print "ID does not exist in Model Repository"
	
	#makes a copy w/ same id
	def copyML(self):
		#self.printModelInformation()
		this_id = raw_input("Enter ID Here --> ")
		print this_id
		if self.verifyID(this_id):
			if self.findAlgorithmBasedOnID(id) is not None:
				print "This model has already been created. . . "
			else:
				self.copy(this_id)
		else:
			print "ID does not exist in Model Repository"

	def copy(self, this_id):
		new_ml = ML_Controller.ML_Controller(self.kb, self.getAlgorithmType(this_id))
		new_ml.copyModel(this_id)
		self.kb.removeModelFromRepository(new_ml.algorithm)
		self.kb.updateDatabaseWithModel(new_ml.algorithm)
		self.kb.addCurrentModel(new_ml.algorithm)
		new_ml.runAlgorithm()
		new_ml.updateDatabaseWithResults()
		self.ml.append(new_ml)
		
	def setupNewML(self):
		models = ['Neural Network', 'Decision Tree', 'SVM']
		possible_choices = range(1, len(models)+1)
		ch_strs = map(str, possible_choices)
		input = ""
		
		while input not in ch_strs:
			print "Pick A Model Type"
			for i in range(0, len(models)):
				print ch_strs[i] + ". " + models[i]
			input = raw_input("--> ")
		input = models[int(input)-1]
		self.createML(input)
		
	def createML(self, input):
		new_ml = ML_Controller.ML_Controller(self.kb, input)
		new_ml.createModel()
		self.kb.updateDatabaseWithModel(new_ml.algorithm)	
		self.kb.addCurrentModel(new_ml.algorithm)
		new_ml.runAlgorithm()
		new_ml.updateDatabaseWithResults()
		self.ml.append(new_ml)
		return new_ml.getID()
		
	def runAlgorithm(self, id=None):
		if id is None:
			id = raw_input("Enter ID of Model --> ")
		model = self.findAlgorithmBasedOnID(id)
		if model is not None:
			res = model.runAlgorithm()
			model.updateDatabaseWithResults()
			return res
		else:	
			print "Model with ID " + id + " does not exist"
		
############################################################################################################
	def optimizeAllModels(self):
		for model in self.ml:
			self.optimizeTask(model.getID())
		
	def optimizeTask(self, id):
		# retrieve model from id
		model = self.findAlgorithmBasedOnID(id)
		if model is not None and not model.isCurrentlyOptimizing: # check to see if optimization flag is true
			print "Adding Model"
			# add to currently optimizing table
			self.addToCurrentlyOptimizingTable(id)
			# set optimization flag to true
			model.isCurrentlyOptimizing = True
			# enqueue to optimization queue
			self.queue.append(model)	
		
	def startOptimization(self):
		# init thread with optimize worker
		if self.queue is not None:
			if len(self.queue) > 0:
				self.stop_event = threading.Event()
				self.optimization_thread = threading.Thread(target=optimizeWorker, args=(self.queue, self.stop_event, self.secs))
				self.optimization_thread.start()
		
	def pauseOptimzation(self):
		# issue stop event and stop thread
		if self.stop_event is not None and self.optimization_thread is not None:
			self.stop_event.set()
			self.optimization_thread.join()
			
	def cancelOptimization(self):
		# issue stop event and stop thread
		self.pauseOptimzation()
		# dequeue through queue setting flags to false
		self.queue.clear()
		for m in self.ml:
			m.isCurrentlyOptimizing = False
			self.removeFromCurrentlyOptimizingTable(m.getID())
			
	def cancelSingleOptimizationTask(self, id):
		self.pauseOptimzation()
		to_remove = None
		for m in self.queue:
			if m.getID() == id:
				to_remove = m
		if to_remove is not None:
			self.queue.remove(to_remove)
			self.removeFromCurrentlyOptimizingTable(id)
		self.startOptimization()
		
	def printInformationOnCurrentlyOptimizingModels(self):
		stmt = "select algorithm_id from CurrentlyOptimizingModels"
		self.kb.executeQuery(stmt)
		row = self.kb.fetchOne()
		current_id = ""
		while row != None:		
			id = row[0]
			self.printModelInformation(id)			
			row = self.kb.fetchOne()
					
	def removeFromCurrentlyOptimizingTable(self,id):
		stmt = "select algorithm_id from CurrentlyOptimizingModels"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		ids = self.kb.fetchAll()
		if (id,) in ids:
			stmt = "delete from CurrentlyOptimizingModels where algorithm_id = " + id
			self.kb.executeQuery(stmt)
			
	def addToCurrentlyOptimizingTable(self, id):
		try:
			stmt = "insert into CurrentlyOptimizingModels(algorithm_id) values (%s)"
			self.kb.executeQuery(stmt,(id,))
		except (MySQLdb.IntegrityError):
			print "Algorithm is already in queue for optimization"
	
############################################################################################################

	def printAlgorithmResults(self):
		#self.pauseOptimzation()
		stmt = "select * from AlgorithmResults"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		print "Algorithm ID\t\t\tAlgorithm Name\t\t\tAccuracy\t\t\tPrecision\t\t\tRecall\t\t\tF1 Score\t\t\t"
		row = self.kb.fetchOne()
		while row != None:
			print "%s\t\t\t%s\t\t\t%s\t\t\t%s\t\t\t%s\t\t\t%s\t\t\t" % (row[0], row[1], row[2], row[3], row[4], row[5])
			row = self.kb.fetchOne()
		#self.startOptimization()
		
	def printModelInformation(self, id=None):
		#self.pauseOptimzation()
		if id is None:
			stmt = "select * from ModelRepository"
		else:
			stmt = "select * from ModelRepository where algorithm_id = " + id
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		row = self.kb.fetchOne()
		current_id = ""
		while row != None:	
			#print row
			if current_id != row[0]:
				print "\nCurrent Algorithm ID: " + row[0] + "\nAlgorithm Type: " + row[1]
				current_id = row[0]
			val = row[3] if row[3] is not None else "None"
			print row[2] + " = " + val
			row = self.kb.fetchOne()
			
		print "\nNo Model Information to Show"
		#self.startOptimization()
		
	def printCurrentModelInformation(self):
		for model in self.ml:
			self.printModelInformation(model.getID())
	
	def checkForCurrentModels(self):
		stmt = "select algorithm_id from CurrentModel"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		row = self.kb.fetchOne()
		i = 0
		while row is not None:
			copy(row[0])
			
	def checkForOptimizingModels(self):
		stmt = "select * from CurrentlyOptimizingModels"
		self.kb.executeQuery(stmt)
		row = self.kb.fetchOne()
		while row is not None:
			id = row[0] #get id
			mdl = self.findAlgorithmBasedOnID(id)
			if mdl is None:
				mdl = self.copy(id)
			# set optimization flag to true
			mdl.isCurrentlyOptimizing = True
			# enqueue to optimization queue
			self.queue.append(mdl)	
		self.startOptimization()	
	
	def runKnowledgeIntegrator(self):
		ki = KnowledgeIntegrator.KnowledgeIntegrator(self.kb, self.ml, self.stacking_classifier, self.other_predictions)
		data = self.kb.getData()
		shuffled_data = shuffle(data)
		splits = numpy.array_split(shuffled_data, 10)
		ki_res = ki.testKI(splits,10,0)

		self.kb.updateDatabaseWithResults(ki)
		
	def splitIntoFolds(self, data, k, seed):
		shuffled_data = shuffle(data, random_state=seed)
		#print shuffled_data
		folds = []
		num_in_folds = len(data) / k
		start = 0
		end = num_in_folds - 1
		for i in range(0,k):
			fold = shuffled_data.iloc[start:end]
			start = end
			end = end + num_in_folds - 1
			#print fold
			folds.append(self.splitIntoXY(fold))
			
		return folds
	
	def getTestTraining(self, curr, others):
		xtest = curr[0]
		ytest = curr[1]
		
		xtrainsets = []
		ytrainsets = []

		for curr in others:
			xtrainsets.append(pandas.DataFrame(curr[0]))
			ytrainsets.append(pandas.DataFrame(curr[1]))

		xtrain = pandas.concat(xtrainsets)
		ytrain = pandas.concat(ytrainsets)
		return xtrain, xtest, ytrain, ytest
	
	def splitIntoXY(self, data):
		#print data
		#print(data.columns.tolist())
		y = data[self.kb.Y] #need to change to reflect varying data...
		#print y
		x = data[self.kb.X]
		#print x	
		return (x,y)
	
	
	def menu(self):
		#TODO
		#1. Create New Model\n
		#2. Recreate Model Based on ID \n
		#3. Create a copy of a model based on ID\n
		#3. Run Model\n => provide ID
		#4. Add model to optimization queue\n => list all current models in queue with current optimization status => have user pick which to add to queue
		#5. Optimize all models => init optimization threads
		#6. Output Model Results (Any current optimization task will be halted and restarted)\n
		#7. View Information on All Model(s)\n => pause all models optimization, print information in modelrepository table
		#8. View Information on Current Model(s)\n => pause all models optimization, print information in current model table where id = current
		#9. Cancel Selected Optimization Task => Print list of models undergoing optimization => Input ID => Cancel Optimization
		#9. Cancel All Optimization Tasks\n => totally cancel all optimization tasks, optimization flags go false
		#10. Quit NEMO\n--> "

		
		options = ['Create New Model', 'Create New Model Based on ID', 'Create a Copy of a Model Based on ID', 'Run Model', 'Run Knowledge Integrator', 'Add Model to Optimization Queue', 'Optimize All Models', 
		'Output All Model Results (Any current optimization task will be halted and restarted)', 'View Information on All Models (Any current optimization task will be halted and restarted)',
		'View Information on Current Models (Any current optimization task will be halted and restarted)', 'View Models in Optimization Queue (Any current optimization task will be halted and restarted)',
		'Cancel Selected Optimization Task', 'Cancel All Optimization Tasks', 'Quit NEMO']
		possible_choices = range(1, len(options)+1)
		ch_strs = map(str, possible_choices)
		input = ""
		while input not in ch_strs:
			print "Main Menu"
			for i in range(0, len(options)):
				print ch_strs[i] + ". " + options[i]
			input = raw_input("--> ")
			
		choice = options[int(input)-1]
		self.processChoice(choice)
		
	def processChoice(self, choice):
		if choice == 'Create New Model':
			self.setupNewML()
		elif choice == 'Create New Model Based on ID':
			self.createModelBasedONID()
		elif choice == 'Create a Copy of a Model Based on ID':
			self.copyML()
		elif choice == 'Run Model':
			self.runAlgorithm()
		elif choice == 'Add Model to Optimization Queue':
			id = raw_input("Enter ID --> ")
			self.optimizeTask(id)
			self.startOptimization()
 		elif choice == 'Optimize All Models':
			self.optimizeAllModels()
			self.startOptimization()
		elif choice == 'Output All Model Results (Any current optimization task will be halted and restarted)':
			self.printAlgorithmResults()
		elif choice == 'View Information on All Models (Any current optimization task will be halted and restarted)': 		
			self.printModelInformation()
		elif choice == 'View Information on Current Models (Any current optimization task will be halted and restarted)':
			self.printCurrentModelInformation()
		elif choice == 'Cancel All Optimization Tasks':
			self.cancelOptimization()
		elif choice == 'Cancel Selected Optimization Task':
			id = raw_input("Enter ID --> ")
			self.cancelSingleOptimizationTask(id)
		elif choice == 'View Models in Optimization Queue (Any current optimization task will be halted and restarted)':
			self.printInformationOnCurrentlyOptimizingModels()
		elif choice == 'Run Knowledge Integrator':
			self.runKnowledgeIntegrator()
			#print "Run KnowledgeIntegrator"
		else:
			self.cancelOptimization()
			sys.exit()
			
def main():
	pid = str(os.getpid())
	dir =  os.path.dirname(os.path.realpath(__file__))
	print dir
	pidfile = "tmp/NEMO.pid"

	if os.path.isfile(pidfile):
		print "%s already exists, exiting" % pidfile
		sys.exit()
	file(pidfile, 'w').write(pid)
	try:
		run(dir)
	finally:
		os.unlink(pidfile)

def run(dir=None):
	if dir is not None:
		nemo = NEMO(dir + "/config/config.json")
	else:
		nemo = NEMO("config/config.json")
	while True:
		nemo.menu()
		
if __name__ == "__main__":
	main()
