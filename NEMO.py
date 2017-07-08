from KnowledgeBase import KnowledgeBase
from ML_Controller import ML_Controller
import MySQLdb
import threading
import sys
import os


#one stop event, pass in the queue and number of seconds to spend optimizing
def optimizeAlgorithmWorker(ml, stp):
	print "Optimizing"
	old_out = sys.stdout
	sys.stdout = open(os.devnull, 'w')
	while not stp.is_set():
		ml.optimizeAlgorithm()
	sys.stdout = old_out
	ml.isCurrentlyOptimizing = False #no thread exists for this model, threads are released when they end
	ml.algorithm.removeModelFromRepository()
	ml.algorithm.updateDatabaseWithModel()
	ml.runAlgorithm()
	ml.updateDatabaseWithResults()
	
class NEMO:
	def __init__(self, filename):
		self.kb = KnowledgeBase(filename)
		self.ml = [] #list of machine learners
		self.threads = [] #list of threads
		#self.event = [] #list of events
		self.checkForCurrentModels()
		
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
		
	#recreates model based on ID
	#NEED method to copy model but create a different ID
	def createModelBasedONID(self):
		#self.printModelInformation()
		id = raw_input("Enter ID Here --> ")
		if self.verifyID(id):
			if self.findAlgorithmBasedOnID(id) is not None:
				print "This model has already been created. . . "
			else:
				self.setupNewML(id)
		else:
			print "ID does not exist in Model Repository"
		
	def setupNewML(self, id=None):
		new_ml = ML_Controller(self.kb)
		new_ml.createModel(id)
		new_ml.runAlgorithm()
		new_ml.updateDatabaseWithResults()
		self.ml.append(new_ml)
	
	def copyML(self):
		#self.printModelInformation()
		this_id = raw_input("Enter ID Here --> ")
		print this_id
		if self.verifyID(this_id):
			new_ml = ML_Controller(self.kb)
			new_ml.copyModel(this_id)
			new_ml.runAlgorithm()
			new_ml.updateDatabaseWithResults()
			self.ml.append(new_ml)
		else:
			print "ID does not exist in Model Repository"

	def runAlgorithm(self):
		#cycle through list of current algorithms to check that id given is legit
		#if so, fetch algorithms, run algorithm
		#else print unsuccessful
		id = raw_input("Enter ID of Model --> ")
		model = self.findAlgorithmBasedOnID(id)
		if model is not None:
			model.runAlgorithm()
			model.updateDatabaseWithResults()
		else:	
			print "Model with ID " + id + " does not exist"
	
	def optimizeAllModels(self):
		for model in self.ml:
			self.createOptimizingWorker(model.getID())
	
	def createOptimizingWorker(self, id):
		#cycle through list of current algorithms to check that id given is legit
		#if so, fetch algorithm
		#else print unsuccessful
		if not self.verifyID(id):
			print "ID does not exist in repository"
			return None
		self.kb.db.commit()	
		model = self.findAlgorithmBasedOnID(id)
		status = model.isCurrentlyOptimizing
		if status == False:
			event = threading.Event()
			thread = threading.Thread(target=optimizeAlgorithmWorker, args=(model, event))
			thread.start()
			self.threads.append({"thread":thread, "event": event, "id": model.getID()})
			self.addToCurrentlyOptimizingTable(model.getID())
		else:
			print "Model is currently being optimized"
	
	def restartOptimizationTasks(self):
		old_threads = self.threads
		self.threads = []
		while len(old_threads) > 0:
			model = old_threads.pop()
			id = model["id"]
			self.createOptimizingWorker(id)
			
	def haltOptimizationTask(self, id):
		for thrd in self.threads:
			if thrd["id"] == id:
				thrd["event"].set()
				thrd["thread"].join()
				return True
		return False
		
	def stopOptimizationTask(self, id):
		to_remove = None
		for thrd in self.threads:
			if thrd["id"] == id:
				thrd["event"].set()
				thrd["thread"].join()
				#self.kb.executeQuery(stmt)
				to_remove = thrd
		if to_remove is not None:
			self.threads.remove(to_remove)
	
	def haltAllOptimizationTasks(self):
		for thrd in self.threads:
			thrd["event"].set()
			thrd["thread"].join()

	def stopAllOptimizationTasks(self):
		while len(self.threads) > 0:
			thrd = self.threads.pop()
			thrd["event"].set()
			thrd["thread"].join()
			#self.kb.executeQuery(stmt)
			#self.kb.db.commit()
			
	def addToCurrentlyOptimizingTable(self, id):
		try:
			stmt = "insert into CurrentlyOptimizingModels(algorithm_id) values (%s)"
			self.kb.executeQuery(stmt,(id,))
		except (MySQLdb.IntegrityError):
			print "Algorithm is already in queue for optimization"
						
	def printInformationOnCurrentlyOptimizingModels(self):
		for model in self.threads:
			self.printModelInformation(model["id"])

	def removeFromCurrentlyOptimizingTable(self,id):
		stmt = "select algorithm_id from CurrentlyOptimizingModels"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		ids = self.kb.fetchAll()
		if (id,) in ids:
			stmt = "delete from CurrentlyOptimizingModels where algorithm_id = " + id
			self.kb.executeQuery(stmt)
			

	def printAlgorithmResults(self):
		stmt = "select * from AlgorithmResults"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		print "Algorithm ID\t\tAlgorithm Name\t\tAccuracy\t\tPrecision\t\tRecall\t\t\tF1 Score\t\tConfusion Matrix"
		row = self.kb.fetchOne()
		while row != None:
			print "%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s" % (row[0], row[1], row[2], row[3], row[4], row[5], row[6])
			row = self.kb.fetchOne()
	
	def printModelInformation(self, id=None):
		if id is None:
			stmt = "select * from ModelRepository"
		else:
			stmt = "select * from ModelRepository where algorithm_id = " + id
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		row = self.kb.fetchOne()
		current_id = ""
		while row != None:		
			if current_id != row[0]:
				print "Current Algorithm ID: " + row[0] + "\nAlgorithm Type: " + row[1]
				current_id = row[0]
			print row[2] + " = " + row[3] + "\n"
			row = self.kb.fetchOne()
			
		print "No Model Information to Show"
		
		
	def printCurrentModelInformation(self):
		for model in self.ml:
			self.printModelInformation(model.getID())
	
	def checkForCurrentModels(self):
		stmt = "select * from CurrentModel"
		self.kb.executeQuery(stmt)
		#self.kb.cursor.execute(stmt)
		rows = self.kb.fetchAll()
		i = 0
		if len(rows) > 0:
			current_row = rows[i]
			current_id = current_row[0]
			self.setupNewML(current_id)
			
			
			
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

		
		options = ['Create New Model', 'Create New Model Based on ID', 'Create a Copy of a Model Based on ID', 'Run Model', 'Add Model to Optimization Queue', 'Optimize All Models', 
		'Output Model Results (Any current optimization task will be halted and restarted)', 'View Information on All Models (Any current optimization task will be halted and restarted)',
		'View Information on Current Models (Any current optimization task will be halted and restarted)', 'Cancel Selected Optimization Task', 'Cancel All Optimization Tasks', 'Quit NEMO']
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
			self.createOptimizingWorker(id)
		elif choice == 'Optimize All Models':
			self.optimizeAllModels()
		elif choice == 'Output Model Results (Any current optimization task will be halted and restarted)':
			self.haltAllOptimizationTasks()
			self.printAlgorithmResults()
			self.restartOptimizationTasks()
		elif choice == 'View Information on All Models (Any current optimization task will be halted and restarted)': 
			self.haltAllOptimizationTasks()
			self.printModelInformation()
			self.restartOptimizationTasks()
		elif choice == 'View Information on Current Models (Any current optimization task will be halted and restarted)':
			self.haltAllOptimizationTasks()
			self.printCurrentModelInformation()
			self.restartOptimizationTasks()
		elif choice == 'Cancel All Optimization Tasks':
			self.stopAllOptimizationTasks()
		elif choice == 'Cancel Selected Optimization Task':
			id = raw_input("Enter ID --> ")
			self.stopOptimizationTask(id)
		else:
			self.stopAllOptimizationTasks()
			sys.exit()
def main():
	nemo = NEMO("config/config.json")
	while True:
		nemo.menu()

def test():
	nemo = NEMO("config/config.json")
	nemo.setupNewML(id="175921957")
	print "Created..."
	
if __name__ == "__main__":
	main()
