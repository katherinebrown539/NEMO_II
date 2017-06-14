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
#	Assumptions: On separate lines, the file contains the MySQL schema for creation of the DATA table
def importData(login_file, data_file, schema_file):
	cursor = connectToDatabase(login_file)
	print cursor
	
def connectToDatabase(login_file):
	fileio = open(login_file, 'r')
	host = fileio.readline()
	port = fileio.readline()
	user = fileio.readline()
	passw = fileio.readline()
	database = fileio.readline()

	db = MySQLdb.connect(host, port, user, passw, database)
	cursor =  db.cursor()
	return cursor
	
def main():
	importData("config/login_template.txt", "data/SPECT.data", "data/SPECT.schema")

if __name__ == "__main__":
	main()