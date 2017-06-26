from KnowledgeBase import KnowledgeBase
from ML_Controller import ML_Controller

		
def main():
	kb = KnowledgeBase("config/login_file.txt")
	kb.importData("data/SPECT.data", "data/SPECT.schema")
	#importData("config/login_file.txt", "data/SPECTF.data", "data/SPECTF.schema")
	ml = ML_Controller(kb)
	
	# try:
    # while True:
        # do_something()
	# except KeyboardInterrupt:
    # pass
	
	ml.runAlgorithm()
	for i in range(0,10):
		ml.optimizeAlgorithm()
if __name__ == "__main__":
	main()
