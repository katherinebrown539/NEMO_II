import MySQLdb
import MySQLdb.cursors
import sys

class KnowledgeBase:
	#method to import data into the MySQL server
	#Preconditions:
	# * login_file - a text file containing the login and database information
	#	Assumptions: On separate lines, the file must contain HOST, PORT, MySQL USER NAME, PASSWORD, DATABASE
	# * data_file - a text file containing the data to be added to the databases
	#	Assumptions: The file is one comma-delimited record per line.
	#				 The first value for the line is the value to classify
	#				 The remaining lines are the attributes
	# * schema_file - a text file containing the MySQL schema for the table
	#	Assumptions: column_name data_type
	#	Assumptions: On separate lines, the file contains the MySQL schema for creation of the DATA table
	def importData(self, data_file, schema_file):
		
		self.cursor.execute("drop table if exists DATA;")
		#db.commit()
		
		#read in schema 
		schema = self.readSchemaFile(schema_file)
		stmt = "create table DATA ( "

		while len(schema) > 1:
			stmt = stmt + schema.pop() + ", "
		
		stmt = stmt + schema.pop() + " );"
		#create new data table
		self.cursor.execute(stmt);
		self.db.commit()
		
		#add new records
		f = open(data_file, 'r')
		for line in f:
			print line
			stmt = "insert into DATA values ( "
			curr_ = line.split(',')
			
			for i in range(0,len(curr_)):
				curr_[i] = curr_[i].strip('\n')

			curr = tuple(curr_)
			print curr
			print len(curr)
			for i in range(0, len(curr)-1):
				stmt = stmt + "%s, "
			stmt = stmt + "%s )"
			print stmt
			self.cursor.execute(stmt, curr)
		self.db.commit()
		#close the database
		self.db.close()
		
	#method to read schema file 
	#Preconditions
	# * schema_file - a text file containing the MySQL schema for the table
	#	Assumptions: column_name data_type
	#	Assumptions: On separate lines, the file contains the MySQL schema for creation of the DATA table
	#Postconditions: Returns list object with schema
	def readSchemaFile(self, schema_file):
		f = open(schema_file, 'r')
		schema = []
		for line in f:
			schema.append(line.strip("\n"))
		f.close()
		schema.reverse()
		return schema

	#Constructor
	#Preconditions:
	# * login_file - a text file containing the login and database information
	#	Assumptions: On separate lines, the file must contain HOST, PORT, MySQL USER NAME, PASSWORD, DATABASE	
	#Postconditions: Connects to database
	def __init__(self, login_file):
		fileio = open(login_file, 'r')
		self.HOST = fileio.readline().strip('\n')
		self.PORT = int(fileio.readline().strip('\n'))
		self.USER = fileio.readline().strip('\n')
		self.PASSWD = fileio.readline().strip('\n')
		self.DATABASE = fileio.readline().strip('\n')
		fileio.close()

		self.db = MySQLdb.connect(host = self.HOST, port = self.PORT, user = self.USER, passwd = self.PASSWD, db = self.DATABASE)
		self.cursor = self.db.cursor()
		
def main():
	kb = KnowledgeBase("config/login_file.txt")
	kb.importData("data/SPECT.data", "data/SPECT.schema")
	#importData("config/login_file.txt", "data/SPECTF.data", "data/SPECTF.schema")

if __name__ == "__main__":
	main()
