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
import traceback
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
		with open(filename) as fd:
			json_data = json.load(fd)

		info = json_data['DATA']
		print info['DATA_JSON']
		self.kbs = self.readInAllDataSources(info['DATA_JSON'], filename)
		self.kb = self.kbs[0]
		self.data_file = info['DATA_JSON']
		self.config_file = filename
		self.ml = [] #list of machine learners
		self.secs = 10
		self.queue = deque()
		self.optimization_thread = None
		self.stop_event = None
		self.checkForCurrentModels()
		self.checkForOptimizingModels()

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
		#print stmt
		self.kb.executeQuery(stmt)
		types = self.kb.fetchOne()
		#print types
		return types[0]

	#same model, different id
	def createModelBasedONID(self):
		#self.printModelInformation()
		id = raw_input("Enter ID Here --> ")
		if self.verifyID(id):
			type = self.getAlgorithmType(id)
			kb = self.getDataSource(id)

			new_ml = ML_Controller.ML_Controller(kb, type)
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

	def copy(self, this_id, new_kb = None):
		algorithm_type = ""
		try:
			algorithm_type = self.getAlgorithmType(this_id)
		except:
			this_id = this_id + "*"
			algorithm_type = self.getAlgorithmType(this_id)
		#getKB
		kb = None
		if new_kb is None:
			kb = self.getDataSource(this_id)
		else:
			kb = new_kb
		self.kb.executeQuery("delete from CurrentModel where algorithm_id="+this_id)
		new_ml = ML_Controller.ML_Controller(kb, algorithm_type)
		new_ml.copyModel(this_id)
		#self.kb.removeModelFromRepository(new_ml.algorithm)
		self.kb.updateDatabaseWithModel(new_ml.algorithm)
		self.kb.addCurrentModel(new_ml.algorithm)
		new_ml.runAlgorithm()
		new_ml.updateDatabaseWithResults()
		self.ml.append(new_ml)
		return new_ml

	def setupNewML(self):
		models = ['Neural Network', 'Decision Tree', 'SVM', 'Random Forest']
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
		kb = self.selectDataSource()
		new_ml = ML_Controller.ML_Controller(kb, input)
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
		else:
			print "Error adding model with ID: " + id

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
		self.pauseOptimzation()
		stmt = "select * from AlgorithmResults"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		print "Algorithm ID\t\tAlgorithm Name\t\tData Source\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1 Score\t\t"
		row = self.kb.fetchOne()
		while row != None:
			#print row
			print "%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s" % (row[0], row[1], row[2], row[3], row[4], row[5],row[6])
			row = self.kb.fetchOne()
		#self.startOptimization()

	def printModelInformation(self, id=None):
		self.pauseOptimzation()
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
		print len(self.ml)
		for model in self.ml:
			self.printModelInformation(model.getID())

	def checkForCurrentModels(self):
		#self.pauseOptimzation()
		stmt = "select algorithm_id from CurrentModel"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		row = self.kb.fetchOne()
		i = 0
		while row is not None:
			self.copy(row[0])
			row = self.kb.fetchOne()
		#self.startOptimization()

	def checkForOptimizingModels(self):
		stmt = "select * from CurrentlyOptimizingModels"
		self.kb.executeQuery(stmt)
		row = self.kb.fetchOne()
		while row is not None:
			id = row[0] #get id
			#print id
			mdl = self.findAlgorithmBasedOnID(id)
			if mdl is None:
				mdl = self.copy(id)
			print "created model"
			# set optimization flag to true
			mdl.isCurrentlyOptimizing = True
			# enqueue to optimization queue
			self.queue.append(mdl)
			row = self.kb.fetchOne()
		#print "Finished checking for models"
		#self.menu()
		self.startOptimization()

	def runKnowledgeIntegrator(self):
		self.pauseOptimzation()
		try:
			#choose data set
			kb = self.selectDataSource()
			mls = []
			#get all mls that use that data set
			for m in self.ml:
				print "kb.name = " + kb.name
				print "m.kb.name = " + m.kb.name
				if kb.name == m.kb.name:
					mls.append(m)
			if len(mls) == 0:
				print "No Models for Data Source Chosen. Please select a different data source"
				return None
			#pass that into the KI.
			ki = KnowledgeIntegrator.KnowledgeIntegrator(kb, mls, self.stacking_classifier, self.other_predictions)
			data = kb.getData()
			shuffled_data = shuffle(data)
			splits = numpy.array_split(shuffled_data, 10)
			ki_res = ki.testKI(splits,10,0)

			self.kb.updateDatabaseWithResults(ki)
			print "Run KnowledgeIntegrator"
		except:
			print "Error running Knowledge Integrator. Please ensure models are created and try again"
			traceback.print_exc()

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

	def splitIntoXY(self, data, kb=None):
		if kb is None:
			kb = self.kb
		#print(data.columns.tolist())
		y = data[kb.Y] #need to change to reflect varying data...
		#print y
		x = data[kb.X]
		#print x
		return (x,y)

	def readInAllDataSources(self, data_json, config):
		kbs = []
		print data_json
		with open(data_json) as fd:
			json_data = json.load(fd)
		print type(json_data)
		print json_data
		print json_data.itervalues()
		for key,val in json_data.iteritems():
			print key + ": " + str(val)
			kbs.append(KnowledgeBase.KnowledgeBase(config, val))
		return kbs

	def printAllDataSources(self):
		print "Data Sources"
		for i in range(0,len(self.kbs)):
			print str(i+1) + ". " + self.kbs[i].name

	def selectDataSource(self):
		possible_choices = range(1, len(self.kbs)+1)
		ch_strs = map(str, possible_choices)
		input = ""
		while input not in ch_strs:
			self.printAllDataSources()
			input = raw_input("--> ")
		#choice = options[int(input)-1]
		to_return = self.kbs[int(input)-1]
		return to_return

	def verifyDataSource(self, data_str):
		sources = []
		for kb in self.kbs:
			sources.append(kb.name)
		return data_str in sources

	def getDataSource(self, id):
		stmt = "select arg_val from ModelRepository where arg_type = \'DATA_SOURCE\' and algorithm_id = " + id
		worked = self.kb.executeQuery(stmt)
		row = self.kb.fetchOne()
		data_name = row[0]
		print data_name
		if self.verifyDataSource(data_name):
			for kb in self.kbs:
				if kb.name == data_name:
					return kb
		#return self.kb

	def getDataSourceFromName(self,name):
		#print(name)
		for kb in self.kbs:
			#print(kb.name)
			if kb.name == name:
				return kb

	def refreshDataSources(self):
		self.cancelOptimization()
		self.kbs = self.readInAllDataSources(self.data_file, self.config_file)
		new_mls = []
		for ml in self.ml:
			kb = self.selectDataSource()
			ml = self.copy(ml.getID(), kb)
			new_mls.append(ml)
			ml.kb.updateDatabaseWithModel(ml.algorithm)
		self.ml = new_mls

	def swapDataSource(self):
		self.cancelOptimization()
		this_id = raw_input("Enter ID Here --> ")
		if self.verifyID(this_id):
			ml = self.findAlgorithmBasedOnID(this_id)
			self.ml.remove(ml)
			kb = self.selectDataSource()
			ml = self.copy(ml.getID(), kb)
			ml.kb.updateDatabaseWithModel(ml.algorithm)
		else:
			print "Invalid ID."

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
		'Cancel Selected Optimization Task', 'Cancel All Optimization Tasks', 'View Available Data Sources', 'Refresh Data Sources (Optimization Will Be Canceled)', 'Swap Data Source on Model (Optimization Will Be Canceled)', 'Quit NEMO']
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
			#self.runKnowledgeIntegrator()
			self.runKnowledgeIntegrator()
			#print "Run KnowledgeIntegrator"
		elif choice == 'View Available Data Sources':
			self.printAllDataSources()
		elif choice == 'Refresh Data Sources (Optimization Will Be Canceled)':
			print "Refesh data sources"
		elif choice == 'Swap Data Source on Model (Optimization Will Be Canceled)':
			self.swapDataSource()
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
