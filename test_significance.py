import NELController
for i in range(1,31):
    print("Iteration #"+str(i))
    output_file = "significance_test/stacked/results_"+str(i)
    NELController("config/facts.json", "config/config.json", output_file)
