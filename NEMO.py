from KnowledgeBase import KnowledgeBase
from ML_Controller import ML_Controller
import threading
import sys
import os

def optimizeAlgorithmWorker(ml, stp):
	print "Optimizing"
	old_out = sys.stdout
	sys.stdout = open(os.devnull, 'w')
	while not stp.is_set():
		ml.optimizeAlgorithm()
	sys.stdout = old_out
	ml.isCurrentlyOptimizing = False #no thread exists for this model, threads are released when they end
	
class NEMO:
	def __init__(self, filename):
		self.kb = KnowledgeBase(filename)
		self.ml = [] #list of machine learners
		self.
		self.thread = [] #list of threads
		self.event = [] #list of events
		self.checkForCurrentModels()
		
		
	def setupNewML(self, id=None):
		new_ml = ML_Controller(self.kb)
		new_ml.createModel(id)
		self.ml.append(new_ml)
		
	def runAlgorithm(self, id):
		#cycle through list of current algorithms to check that id given is legit
		#if so, fetch algorithms, run algorithm
		#else print unsuccessful
		self.ml.runAlgorithm()
			
	
	def optimizeAlgorithm(self, id):
		#cycle through list of current algorithms to check that id given is legit
		#if so, fetch algorithm
		#else print unsuccessful
		
		#check to see if algorithm's optimization flag is true
		#if true, thread is in optimizing queue
		#else, thread is not in optimizing queue
		self.event = threading.Event()
		self.thread = threading.Thread(target=optimizeAlgorithmWorker, args=(self.ml, self.event))
		self.thread.start()
		
	def printAlgorithmResults(self):
		stmt = "select * from AlgorithmResults"
		self.kb.cursor.execute(stmt)
		print "Algorithm ID\t\tAlgorithm Name\t\tAccuracy\t\tPrecision\t\tRecall\t\t\tF1 Score\t\tConfusion Matrix"
		row = self.kb.cursor.fetchone()
		while row != None:
			print "%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s" % (row[0], row[1], row[2], row[3], row[4], row[5], row[6])
			row = self.kb.cursor.fetchone()
	
	def printAlgorithmInformation(self, id=None):
		if id is None:
			stmt = "select * from ModelRepository"
		else:
			stmt = "select * from ModelRepository where algorithm_id = " + id
		self.kb.cursor.execute(stmt)
		row = self.kb.cursor.fetchone()
		current_id = ""
		while row != None:		
			if current_id != row[0]:
				print "Current Algorithm ID: " + row[0] + "\nAlgorithm Type: " + row[1]
				current_id = row[0]
			print row[2] + " = " + row[3] + "\n"
			row = self.kb.cursor.fetchone()
			
		print "No Model Information to Show"
		
	def checkForCurrentModels(self):
		stmt = "select * from CurrentModel"
		self.kb.cursor.execute(stmt)
		rows = self.kb.cursor.fetchall()
		i = 0
		if len(rows) > 0:
			current_row = rows[i]
			current_id = current_row[0]
			self.setupNewML(current_id)
			
			
			
	def menu(self):
		#TODO
		#1. Create New Model\n
		#2. Create Model Based on ID \n
		#3. Run Model\n => provide ID
		#4. Add model to optimization queue\n => list all current models in queue with current optimization status => have user pick which to add to queue
		#5. Optimize all models => init optimization threads
		#6. Output Model Results (Any current optimization task will be halted and restarted)\n
		#7. View Information on All Model(s)\n => pause all models optimization, print information in modelrepository table
		#8. View Information on Current Model(s)\n => pause all models optimization, print information in current model table where id = current
		#9. Cancel Selected Optimization Task => Print list of models undergoing optimization => Input ID => Cancel Optimization
		#9. Cancel All Optimization Tasks\n => totally cancel all optimization tasks, optimization flags go false
		#10. Quit NEMO\n--> "

		
		options = ['Create New Model', 'Create New Model Based on ID', 'Run Model', 'Add Model to Optimization Queue', 'Optimize All Models', 
		'Output Model Results (Any current optimization task will be halted and restarted)', 'View Information on All Models (Any current optimization task will be halted and restarted)',
		'View Information on Current Models (Any current optimization task will be halted and restarted)', 'Cancel All Optimization Tasks', 'Quit NEMO']
		possible_choices = range(1, len(options)+1)
		ch_strs = map(str, possible_choices)
		input = ""
		while input not in ch_strs:
			print "Main Menu"
			for i in range(0, len(options)):
				print ch_strs[i] + ". " + options[i]
			input = raw_input("--> ")
			
	
		
def main():
	nemo = NEMO("config/config.json")
	#while True:
	nemo.menu()

def test():
	nemo = NEMO("config/config.json")
	nemo.setupNewML(id="175921957")
	print "Created..."
	
if __name__ == "__main__":
	main()
