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
	
class NEMO:
	def __init__(self, filename):
		self.kb = KnowledgeBase(filename)
		self.ml = None #will be a list
		self.thread = None #will be a list
		self.event = None #will be a list
		self.checkForCurrentModels()
		
		
	def setupNewML(self, id=None):
		self.ml = ML_Controller(self.kb)
		self.ml.createModel(id)
		
	def runAlgorithm(self):
		self.ml.runAlgorithm()
			
	
	def optimizeAlgorithm(self):
		self.event = threading.Event()
		self.thread = threading.Thread(target=optimizeAlgorithmWorker, args=(self.ml, self.event))
		#self.thread.daemon = False
		self.thread.start()
		
	def printAlgorithmResults(self):
		stmt = "select * from AlgorithmResults"
		self.kb.cursor.execute(stmt)
		print "Algorithm ID\t\tAlgorithm Name\t\tAccuracy\t\tPrecision\t\tRecall\t\t\tF1 Score\t\tConfusion Matrix"
		row = self.kb.cursor.fetchone()
		while row != None:
			print "%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s" % (row[0], row[1], row[2], row[3], row[4], row[5], row[6])
			row = self.kb.cursor.fetchone()
	
	def printAlgorithmInformation(self):
		stmt = "select * from ModelRepository"
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
		menu = "Main Menu:\n1. Create New Model\n2. Run New Model\n3. Optimize the model\n4. Output Model Results (Any current optimization task will be halted and restarted)\n5. View Model Information\n6. Cancel All Optimization Tasks\n7. Quit NEMO\n--> "
		choices = ["1","2","3","4","5","6","7"]
		choice = ""
		while choice not in choices:
			choice = raw_input(menu)
		if choice == "1":
			self.setupNewML()
		elif choice == "2":
			self.runAlgorithm()
		elif choice == "3":
			self.optimizeAlgorithm()
		elif choice == "4":
			if self.event is not None and self.thread is not None:
				self.event.set()
				self.thread.join()
			self.printAlgorithmResults()
			if self.thread is not None and self.event is not None: #there was a thread that was stopped...
				self.optimizeAlgorithm()
		elif choice == "5":
			#id = raw_input("Algorithm ID --> ")
			self.printAlgorithmInformation()
		elif choice == "6":
			self.event.set()
			self.thread.join()
		else:
			#set up forking
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
