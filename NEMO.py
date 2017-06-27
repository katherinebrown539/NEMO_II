from KnowledgeBase import KnowledgeBase
from ML_Controller import ML_Controller
import sys
		
def main():
	kb = KnowledgeBase("config/login_file.txt")
	kb.importData("data/SPECT.data", "data/SPECT.schema")
	#importData("config/login_file.txt", "data/SPECTF.data", "data/SPECTF.schema")
	ml = ML_Controller(kb)
	ml.runAlgorithm()
	
	try:
		while True:
			ml.optimizeAlgorithm()
	except:
			print "NEMO ending. . ."
			sys.exit()
	
	
if __name__ == "__main__":
	main()
