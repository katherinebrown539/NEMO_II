from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller, KnowledgeIntegrator
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
		self.stacking_classifier = info["STACKER"]
		self.experiment = "self.experiment"+str(info['EXPERIMENT'])+"()"
	
		
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
		self.ki = KnowledgeIntegrator.KnowledgeIntegrator( self.kb, self.models, self.stacking_classifier)
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
		stack_res = []
		for partition in self.partitions:
			# #split 10/90
			splits = numpy.array_split(partition, 10)
			num = self.testKI(self.ki, splits)
			stack_res.append(num)
		print stack_res
		mean = numpy.mean(stack_res)
		std_ = numpy.std(stack_res)
		stack_res.append(mean)
		stack_res.append(std_)
		print stack_res
		res[self.ki.getName()] = stack_res
		print res
		print ""
		return res
			
	def testKI(self, ki, splits):
		holdout = splits.pop()
		remain = pandas.concat(splits)
		folded_data = deque(self.splitIntoFolds(remain,self.num_folds,self.random_seed))
		folds = []
		for i in range(0, self.num_folds):
			curr = folded_data.popleft()
			info = self.getTestTraining(curr, folded_data)
			folds.append(info)
			folded_data.append(curr)
		#print len(folds)
		ki.trainAndCreateMetaDataSet(folds)
		ki.trainMetaModel()
		xtrain, ytrain = self.splitIntoXY(remain)
		fold = (xtrain, None, ytrain, None)
		ki.trainLevelOneModels(fold)
		curr_res = ki.runModel(holdout)
		return curr_res.get(self.metric)
		
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
		
		
		
def main():
	exp = NEMO_Experiment('config/experiment.json')
	exp.setUpExperiment()
	#exp.experiment()
	
	
if __name__ == '__main__':
	main()