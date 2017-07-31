from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller, KnowledgeIntegrator, SemiSupervisedController
from collections import deque
from NEMO import NEMO
import pandas
import numpy
from pandas import DataFrame
from sklearn.utils import shuffle
import pandas.io.sql as psql
import MySQLdb
import sys
import os
import time
import random
import json
#test comment for git
class NEMO_Experiment:
	def __init__(self, filename):
		self.NEMO_instance = NEMO(filename)
		self.kb = self.NEMO_instance.kb
		self.readExperimentFile(filename)
		
		
	def getData(self):
		stmt = "select * from DATA"
		return pandas.read_sql_query(stmt, self.kb.db)
		
	def splitIntoFolds(self, data, k, seed):
		shuffled_data = shuffle(data, random_state=seed)
		#print shuffled_data
		folds = []
		num_in_folds = len(data) / k
		start = 0
		end = num_in_folds - 1
		for i in range(0,k):
			fold = shuffled_data.iloc[start:end]
			start = end
			end = end + num_in_folds - 1
			#print fold
			folds.append(self.splitIntoXY(fold))
			
		return folds
		
	def splitIntoXY(self, data):
		#print data
		#print(data.columns.tolist())
		y = data[self.kb.Y] #need to change to reflect varying data...
		#print y
		x = data[self.kb.X]
		#print x	
		return (x,y)
	
	def readExperimentFile(self, filename):
		with open(filename) as fd:
			json_data = json.load(fd)
			
		info = json_data['EXPERIMENT']
		self.algorithms = info['ALGORITHMS']
		self.num_folds = info['NUM_FOLDS']
		self.random_seed = info['RANDOM_SEED'] if info['RANDOM_SEED'] != "None" else None
		self.output_file = info['OUTPUT_FILE']
		self.num_partitions = info['NUM_PARTITIONS']
		self.metric = info['METRIC']
		self.experiment = "self.experiment"+str(info['EXPERIMENT'])+"()"
		info = json_data['KNOWLEDGE_INTEGRATOR']
		self.stacking_classifier = info["STACKER"]
		self.use_features = info["USE_FEATURES"] == "True"
		self.other_predictions = info["OTHER_PREDICTIONS"] if info['OTHER_PREDICTIONS'] != "None" else None
		info = json_data['SEMI_SUPERVISED']
		self.split_method = info["SPLIT_METHOD"]
		self.num_train = info["k"]
		
		
	def getTestTraining(self, curr, others):
		xtest = curr[0]
		ytest = curr[1]
		
		xtrainsets = []
		ytrainsets = []

		for curr in others:
			xtrainsets.append(pandas.DataFrame(curr[0]))
			ytrainsets.append(pandas.DataFrame(curr[1]))

		xtrain = pandas.concat(xtrainsets)
		ytrain = pandas.concat(ytrainsets)
		return xtrain, xtest, ytrain, ytest
	
	def kfoldcv(self, model, data, k, random_seed, metric):
		folded_data = deque(self.splitIntoFolds(data,k,random_seed))
		scores = []
		curr = None
		for i in range(0,k):
			curr = folded_data.popleft()
			xtrain, xtest, ytrain, ytest = self.getTestTraining(curr, folded_data)
			other_train = None
			other_test = None
			if self.other_predictions is not None:
				split = self.splitIntoAttributesOther(xtrain)
				xtrain = split[0]
				other_train = split[1]
				split = self.splitIntoAttributesOther(xtest)
				xtest = split[0]
				other_test = split[1]
			model.createModelPreSplit(xtrain,xtest,ytrain,ytest)
			model.runAlgorithm()
			scores.append(model.algorithm.results[metric])
			folded_data.append(curr)
		#print model.algorithm.algorithm_name + ": " + str(scores)
		return numpy.mean(scores)
		
	def setUpExperiment(self):
		data = self.getData()
		self.data = shuffle(data, random_state = self.random_seed)
		self.partitions = numpy.array_split(data, self.num_partitions)
		self.models = []
		for mdl in self.algorithms:
			self.models.append(ML_Controller.ML_Controller(self.kb, mdl))
		self.ki = KnowledgeIntegrator.KnowledgeIntegrator( self.kb, self.models, self.stacking_classifier, self.other_predictions, self.use_features)
		self.semi_supervised = SemiSupervisedController.SemiSupervisedController(self.models, self.kb, self.split_method, self.num_train)
		res = eval(self.experiment)
		self.writeToCSV(res, self.output_file)
		
	def experiment1(self):
		exp_res = {}
		for mdl in self.models:
			mdl_res = []
			for partition in self.partitions:
				score = self.kfoldcv(mdl, partition, self.num_folds, self.random_seed, self.metric)
				mdl_res.append(score)
			mean = numpy.mean(mdl_res)
			std_ = numpy.std(mdl_res)
			mdl_res.append(mean)
			mdl_res.append(std_)
			exp_res[mdl.algorithm.algorithm_name] = mdl_res
		#self.writeToCSV1(exp_res, self.output_file)
		print exp_res
		print ""
		return exp_res
		#print exp_res
		
	def experiment2(self):
		#complete experiment 1
		res = self.experiment1()
		# res = {}
		print "res: " + str(res)
		stack_res = []
		for partition in self.partitions:
			# #split 10/90
			splits = numpy.array_split(partition, 10)
			#ki_res = self.testKI(self.ki,splits)
			ki_res = self.ki.testKI(splits, self.num_folds, self.random_seed)
			num = ki_res.get(self.metric)
			print self.metric + " = " + str(num)
			stack_res.append(num)
		# print stack_res
		mean = numpy.mean(stack_res)
		std_ = numpy.std(stack_res)
		stack_res.append(mean)
		stack_res.append(std_)
		# print stack_res
		res[self.ki.getName()] = stack_res
		print ""
		return res
	
	def experiment3(self):
		res = {}
		for classifier in self.models:
			res[classifier.getName()] = []
		res[self.ki.getName()] = []
		count = 1
		for partition in self.partitions:
			print "partition: " + str(count)
			self.semi_supervised.split(seed = self.random_seed, data = partition)
			self.semi_supervised.trainClassifiers(self.stacking_classifier, self.other_predictions, self.use_features)
			part_res = self.semi_supervised.testClassifiers(self.metric, self.random_seed, self.num_folds)
			
			for classifier in self.models:
				res[classifier.getName()].append(part_res.get(classifier.getName()))
			res[self.ki.getName()].append(part_res.get(self.ki.getName()))
			count = count+1
		return res	
		
	def writeToCSV(self, results, filename):
		file = open(filename, 'w')
		line = 'Algorithm,'
		for i in range(1,len(self.partitions)):
			line += 'partition_'+str(i)+","
		line += "partition_"+str(len(self.partitions)) +",mean,std. deviation"+ "\n"
		print line
		file.write(line)
		for key,val in results.iteritems():
			file.write( key+","+(','.join(map(str, val)))+"\n")
		file.close()
		
	def splitIntoAttributesOther(self, data):
		if data is not None:
			atr = list(set(self.kb.X) - set(self.other_predictions))
			x = data[atr]
			other = data[self.other_predictions]
			return(x,other)
		else:
			return (None, None)

def main():
	exp = NEMO_Experiment('config/experiment.json')
	exp.setUpExperiment()
	#exp.experiment()
	
	
if __name__ == '__main__':
	main()