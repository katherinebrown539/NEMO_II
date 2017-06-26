from KnowledgeBase import KnowledgeBase
from ML_Controller import ML_Controller

		
def main():
	kb = KnowledgeBase("config/login_file.txt")
	kb.importData("data/SPECT.data", "data/SPECT.schema")
	#importData("config/login_file.txt", "data/SPECTF.data", "data/SPECTF.schema")
	ml = ML_Controller(kb)
	
	#for i in range(0,10):
	#while True:
	#	ml.runAlgorithm()
	ml.runAlgorithm()
	
if __name__ == "__main__":
	main()
