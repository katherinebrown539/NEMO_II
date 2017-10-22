import MySQLdb
import MySQLdb.cursors
import sys
import pandas
import json
import pandas
##############################################################################################################
# KnowledgeBase class																						 #
# Prototype KnowledgeBase for NEMO																			 #
##############################################################################################################
# ****** INSTANCE VARIABLES ******                                                                           #
# HOST     -     host for the database (typically localhost), STRING                                         #
# PORT     -     Port number for database, MUST BE AN INTEGER!!                                              #
# USER     -     User name for database, STRING                                                              #
# PASSWD   -     Password for database, STRING                                                               #
# DATABASE -     Database name, STRING                                                                       #
# db       -     object connected to the database, DATABASE CONNECTION OBJECT                                #
# cursor   -     cursor that executes MySQL commands, CURSOR OBJECT                                          #
# X        -     attribute names, LIST of STRINGS                                                            #
# Y        -     target name, STRING					                                                     #
# schema   -     list containing MySQL compatible schema, format: ATTR_NAME ATTR_TYPE, LIST OF STRINGS       #
#                                                                                                            #
###############################################################################################################

class KnowledgeBase:
	#method to import data into the MySQL server
	#Preconditions:
	# * KnowledgeBase object is created with connection to db established
	# * data_file - a text file containing the data to be added to the databases
	#	Assumptions: The file is one comma-delimited record per line.
	#				 The first value for each line is the value to classify
	#				 The remaining values on each line are the attributes
	# * schema_file - a text file containing the MySQL schema for the table
	#	Assumptions: column_name data_type
	#	Assumptions: On separate lines, the file contains the MySQL schema for creation of the DATA table
	def importData(self, data_file, schema_file):

		self.cursor.execute("drop table if exists " + self.name + ";")
		#db.commit()

		#read in schema
		self.readSchemaFile(schema_file)
		stmt = "create table " + self.name + " ( "

		while len(self.schema) > 1:
			stmt = stmt + self.schema.pop() + ", "

		stmt = stmt + self.schema.pop() + " );"
		#create new data table
		self.cursor.execute(stmt);
		self.db.commit()

		#add new records
		f = open(data_file, 'r')
		for line in f:
			#print line
			stmt = "insert into " + self.name + " values ( "
			#print stmt
			curr_ = line.split(',')

			for i in range(0,len(curr_)):
				curr_[i] = curr_[i].strip('\n')

			curr = tuple(curr_)
			#print curr
			#print len(curr)
			for i in range(0, len(curr)-1):
				stmt = stmt + "%s, "
			stmt = stmt + "%s )"
			#print stmt
			self.cursor.execute(stmt, curr)
		self.db.commit()
		#close the database

		f.close()

	#method to read schema file
	#Preconditions
	# * schema_file - a text file containing the MySQL schema for the table
	#	Assumptions: column_name data_type
	#	Assumptions: On separate lines, the file contains the MySQL schema for creation of the DATA table
	#Postconditions: Returns list object with schema
	def readSchemaFile(self, schema_file):
		f = open(schema_file, 'r')
		self.schema = []
		for line in f:
			self.schema.append(line.strip("\n"))
		f.close()
		self.schema.reverse()
		self.getXYTokens()
		#return schema

	#method to get a list of names from the attributes and targets
	#Preconditions:
	# * schema has already been read from file (ie readSchemaFile has already been called)
	#Postconditions: self.X has names of the attributes, self.Y has the names of the target
	def getXYTokens(self):
		self.X = []
		for i in range(0,len(self.schema)):
			tokens = self.schema[i].split(' ')
			if(tokens[0] != self.Y):
				self.X.append(tokens[0])
		self.X.reverse()
		#tokens = self.schema[len(self.schema)-1].split(' ')
		#self.Y = tokens[0]


	def updateDatabaseWithResults(self, algorithm):
		# results = (algorithm.results['ID'], algorithm.results['Name'], algorithm.results['Accuracy'],  algorithm.results['Precision'], algorithm.results['Recall'], algorithm.results['F1'], str(algorithm.results['Confusion_Matrix']).replace('\n', ""))
		# stmt = "insert into AlgorithmResults(algorithm_id, algorithm_name, accuracy, prec, recall, f1, confusion_matrix) values (%s,%s,%s,%s,%s,%s,%s)"

		results = (algorithm.results['ID'], algorithm.results['Name'], self.name, algorithm.results['Accuracy'],  algorithm.results['Precision'], algorithm.results['Recall'], algorithm.results['F1'])
		stmt = "insert into AlgorithmResults(algorithm_id, algorithm_name, data_source, accuracy, prec, recall, f1) values (%s,%s,%s,%s,%s,%s,%s)"
		# print stmt
		# print str(results)
		self.executeQuery(stmt, results)

	# def getData(self):
	# 	stmt = "select * from " + self.name
	# 	return pandas.read_sql_query(stmt, self.db)

	#Constructor
	#Preconditions:
	# * login_file - a text file containing the login and database information
	#	Assumptions: On separate lines, the file must contain HOST, PORT, MySQL USER NAME, PASSWORD, DATABASE
	#Postconditions: Connects to database
	def __init__(self, config_file, file_info_dict=None, copy=False):
		self.config_file = config_file
		self.file_info_dict = file_info_dict
		with open(config_file) as fd:
			json_data = json.load(fd)

		info = json_data['DATABASE']
		self.HOST = info['HOST']
		self.PORT = int(info['PORT'])
		self.USER = info['USER']
		self.PASSWD = info['PASS']
		self.DATABASE = info['DB']


		self.db = MySQLdb.connect(host = self.HOST, port = self.PORT, user = self.USER, passwd = self.PASSWD, db = self.DATABASE, use_unicode=True, charset="utf8")
		self.cursor = self.db.cursor()


		file_info = file_info_dict if file_info_dict is not None else json_data['DATA']

		self.name = file_info["DATA_NAME"]
		self.schema = None
		self.X = None
		self.Y = file_info['CLASS']
		print (self.name)
		print (file_info['DATA'])
		print (file_info['SCHEMA'])
		print (self.Y)
		self.multi = file_info['MULTI-CLASS'] == "True"
		if not copy:
			self.importData(file_info['DATA'], file_info['SCHEMA'])
		while(self.X.count(self.Y) > 0):
			self.X.remove(self.Y)

	def copy(self):
		self.cursor.close()
		new_kb = KnowledgeBase(self.config_file, self.file_info_dict, copy=True)
		self.cursor = self.db.cursor()
		return new_kb

	def executeQuery(self, query, args=None):
		#self.connect()
		complete = False
		print (query)
		#if args is not None: print args
		while not complete:
			try:
				if args is None:
					self.cursor.execute(query)
					self.db.commit()
				else:
					self.cursor.execute(query, args)
			except (AttributeError, MySQLdb.OperationalError):
				self.connect()
			complete = True
			self.db.commit()
		#self.disconnect()

	def fetchOne(self):
		complete = False
		#self.connect()
		while not complete:
			try:
				to_return =  self.cursor.fetchone()
				complete = True
				return to_return
			except (AttributeError, MySQLdb.OperationalError):
				self.db = MySQLdb.connect(host = self.HOST, port = self.PORT, user = self.USER, passwd = self.PASSWD, db = self.DATABASE)
				self.cursor = self.db.cursor()

		#self.disconnect()

	def fetchAll(self):
		#self.connect()
		complete = False
		while not complete:
			try:
				return self.cursor.fetchall()
			except (AttributeError, MySQLdb.OperationalError):
				self.db = MySQLdb.connect(host = self.HOST, port = self.PORT, user = self.USER, passwd = self.PASSWD, db = self.DATABASE)
				self.cursor = self.db.cursor()
			complete=True
		#self.disconnect()

	def removeModelFromRepository(self, model):
		stmt = "delete from ModelRepository where algorithm_id = " + model.algorithm_id
		self.executeQuery(stmt)

	def updateDatabaseWithModel(self, model):
		#check to see if model is in database
		#if so, add modified char to id (mod_id)
		# update ModelRepository set algorithm_id = mod_id where algorithm_id = model.algorithm_id
		stmt = "select * from ModelRepository where algorithm_id = " + model.algorithm_id
		self.executeQuery(stmt)
		row = self.fetchOne()
		mod_id = None
		if row is not None: #it exists in the database
			mod_id = model.algorithm_id + "*"
			stmt = "update ModelRepository set algorithm_id = \'" + mod_id + "\' where algorithm_id = \'" + model.algorithm_id + "\'"
			#print stmt
			self.executeQuery(stmt)
		arguments = model.get_params()
		#print arguments
		stmt = "insert into ModelRepository (algorithm_id, algorithm_name, arg_type, arg_val) values ( %s, %s, %s, %s)"
		args =  (model.algorithm_id, model.algorithm_name, "DATA_SOURCE", self.name)
		self.executeQuery(stmt, args)
		for key, value in arguments.iteritems():
			#print key + ": " + str(value)
			stmt = "insert into ModelRepository (algorithm_id, algorithm_name, arg_type, arg_val) values ( %s, %s, %s, %s)"
			args = (model.algorithm_id, model.algorithm_name, key, str(value))
			self.executeQuery(stmt, args)
		#remove backup model...
		if mod_id is not None: #we had to use mod_id
			stmt = "delete from ModelRepository where algorithm_id = \'" + mod_id + "\'"
			#print stmt
			self.executeQuery(stmt)
		self.db.commit()

	def addCurrentModel(self, model):
		stmt = "insert into CurrentModel(algorithm_id) values (%s)"
		args = (model.algorithm_id,)
		self.executeQuery(stmt, args)

	def removeCurrentModel(self, model):
		stmt = "delete from CurrentModel where algorithm_id = " + model.algorithm_id
		self.executeQuery(stmt)

	def getData(self):
		#self.connect()
		#stmt = "select * from " + self.name
		stmt = "select " + ",".join(self.X ) + ", " + self.Y + " from " + self.name
		print (stmt)
		to_return =  pandas.read_sql_query(stmt, self.db)
		print("self.X = " + str(self.X))
		print("INDEX =  " + str(to_return.columns))
		print (to_return.head(20))
		self.X = list(to_return.columns)
		#self.disconnect()
		return to_return

	def connect(self):
		self.db = MySQLdb.connect(host = self.HOST, port = self.PORT, user = self.USER, passwd = self.PASSWD, db = self.DATABASE)
		self.cursor = self.db.cursor()

	def disconnect(self):
		self.db.commit()
		self.db.close()

	#DESTRUCTOR
	#commits all changes to database and closes the connection
	def __del__(self):
		self.db.commit()
		self.db.close()

############################################################################################################################################
# 	END OF KNOWLEDGE BASE CLASS 																										   #
############################################################################################################################################

######## Executable For Testing #########
def main():
	kb = KnowledgeBase("config/config.json")


if __name__ == "__main__":
	main()
