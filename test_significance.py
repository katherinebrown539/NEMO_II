from NELController import NELController
for i in range(0,15):
    print("Iteration #"+str(i+1))
    output_file = "significance_test/deep/results_"+str(i+1)+".csv"
    NELController("config/facts.json", "config/config.json", output_file)
