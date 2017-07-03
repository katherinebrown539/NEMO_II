from KnowledgeBase import KnowledgeBase
from ML_Controller import ML_Controller
import sys
	
class NEMO:
	def __init__(self, filename):
		self.kb = KnowledgeBase(filename)
		self.ml = None #will be a list
		self.thread = None #will be a list
		self.event = None #will be a list
		
	def setupNewML(self):
		self.ml = ML_Controller(self.kb)
		#return ml
		
	def runAlgorithm(self):
		self.ml.runAlgorithm()
		
	def optimizeAlgorithm(self):
		self.ml.optimizeAlgorithm()

	def menu():
		menu = "Main Menu:\n1. Create New Model\n2. Run New Model\n3. Optimize the model\n4. Output Model Results (Any current optimization task will be halted)\n5. Cancel All Optimization Tasks\n6. Quit NEMO"
		choices = ["1","2","3","4","5","6"]
		choice = ""
		while choice not in choices:
			choice = raw_input(menu)
		if choice == "1":
			self.setUpNewML()
		elif choice == "2":
			self.runAlgorithm()
		elif choice == "3":
			print "Need to set up multi-threading for background optimization"
			self.optimizeAlgorithm()
		elif choice == "4":
			print "Print out algorithm results here"
		elif choice == "5":
			print "When multi-threading background optimization is setup, this will stop any optimization tasks"
		else:
			sys.exit()
		
def main():
	nemo = NEMO("config/config.json")
	while True:
		nemo.menu()

def old_main():
	nemo = NEMO("config/config.json")
	ml = nemo.setupNewML()
	
	nemo.runAlgorithm()
	
	try:
		while True:
			try:
				nemo.optimizeAlgorithm()
			except:
				sys.exit()
	except:
			print "NEMO ending. . ."
			sys.exit()
	
	
if __name__ == "__main__":
	main()
