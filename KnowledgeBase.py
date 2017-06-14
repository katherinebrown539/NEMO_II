import MySQLdb
import MySQLdb.cursors
import sys
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
def importData(login_file, data_file, schema_file):
	db = connectToDatabase(login_file)
	cursor = db.cursor()
	#delete if exists DATA
	cursor.execute("drop table if exists DATA;")
	#db.commit()
	
	#read in schema 
	schema = readSchemaFile(schema_file)
	stmt = "create table DATA ( "

	while len(schema) > 1:
		stmt = stmt + schema.pop() + ", "
	
	stmt = stmt + schema.pop() + " );"
	#create new data table
	cursor.execute(stmt);
	db.commit()
	
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
		cursor.execute(stmt, curr)
	db.commit()
	#close the database
	db.close()
	
#method to read schema file 
#Preconditions
# * schema_file - a text file containing the MySQL schema for the table
#	Assumptions: column_name data_type
#	Assumptions: On separate lines, the file contains the MySQL schema for creation of the DATA table
#Postconditions: Returns list object with schema
def readSchemaFile(schema_file):
	f = open(schema_file, 'r')
	schema = []
	for line in f:
		schema.append(line.strip("\n"))
	f.close()
	schema.reverse()
	return schema

#method to connect the MySQL database
#Preconditions:
# * login_file - a text file containing the login and database information
#	Assumptions: On separate lines, the file must contain HOST, PORT, MySQL USER NAME, PASSWORD, DATABASE	
#Postconditions: Returns the connection to the database
def connectToDatabase(login_file):
	fileio = open(login_file, 'r')
	host_ = fileio.readline().strip('\n')
	port_ = fileio.readline().strip('\n')
	user_ = fileio.readline().strip('\n')
	passw_ = fileio.readline().strip('\n')
	database_ = fileio.readline().strip('\n')
	fileio.close()

	db = MySQLdb.connect(host = host_, port = int(port_), user = user_, passwd = passw_, db = database_)
	return db
	
def main():
	importData("config/login_file.txt", "data/SPECT.data", "data/SPECT.schema")

if __name__ == "__main__":
	main()
