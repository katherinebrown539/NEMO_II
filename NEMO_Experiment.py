from KnowledgeBase import KnowledgeBase
from Classifiers import ML_Controller
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
		
	def getTestTraining(self, curr, others):
		xtest = curr[0]
		ytest = curr[1]
		
		xtrainsets = []
		ytrainsets = []

		for curr in others:
			xtrainsets.append(curr[0])
			ytrainsets.append(curr[1])
			
		xtrain = pandas.concat(xtrainsets)
		ytrain = pandas.concat(ytrainsets)
		
		return xtrain, xtest, ytrain, ytest
	
	def setUpExperiment(self):
		data = self.getData()
		data = shuffle(data, random_state = self.random_seed)
		self.partitions = numpy.array_split(data, self.num_partitions)
		self.models = []
		for mdl in self.algorithms:
			self.models.append(ML_Controller.ML_Controller(self.kb, mdl))
		
	
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
		
	def experiment(self):
		exp_res = {}
		for mdl in self.models:
			mdl_res = []
			for partition in self.partitions:
				score = self.kfoldcv(mdl, partition, self.num_folds, self.random_seed, self.metric)
				mdl_res.append(score)
			mean = numpy.mean(mdl_res)
			mdl_res.append(mean)
			std_ = numpy.std(mdl_res)
			mdl_res.append(std_)
			exp_res[mdl.algorithm.algorithm_name] = mdl_res
		self.writeToCSV(exp_res, self.output_file)
		print exp_res
		
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
	exp = NEMO_Experiment('config/experiment1.json')
	exp.setUpExperiment()
	exp.experiment()
	
	
if __name__ == '__main__':
	main()