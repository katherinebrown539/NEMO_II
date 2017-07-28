# Classifiers for NEMO

NEMO can support classifiers from popular libraries such as SciKit-Learn or source code. To be compatible with the explorer and experimenter applications; however, models need to follow the followign interface:<br/>
1. createModelFromID - Given an id, this method pulls the attributes from the MySQL database, parses them to set the parameters, and creates the model with a new ID. <br/>
2. copyModel - Given an id, this method pulls the attributes from the MySQL database, parses them to set the parameters, and creates the model with the same ID. <br/>
3. createModelPreSplit - Given test and training sets for attributes and targets and optional dictionary of attributes, this method creates the model, designates the test/training sets, and performs an initial fitting to the training sets.
4. createModel - Given attributes and targets, this method creates the models, splits into training and test sets, and performs initial fitting to the training set. <br/>
5. runModel - This method runs the model on the pre-provided test set and returns a dictionary with the model id, model name, accuracy, precision, recall, f1 score and confusion matrix.
6. predict - This method predicts the labels for a given attribute set
7. optimize - Given a metric and optimization method (only coordinate ascent is supported), this method calls and returns the optimized model.
8. coordinateAscent - This method controls the coordinate ascent optimization for the model.
