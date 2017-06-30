# NEMO_II
NEMO_II Data Requirements

Currently, NEMO can import data from flat text files and load them into a MySQL database.

To do this, NEMO must be provided two text files. These are best stored here.


# Data File Requirements
1) There can be no missing data. All the attributes must have some value
2) The class must be specified in config/config.json. If the class specified does not exist or spelled incorrectly, the program will not execute. 
3) The file must be comma-delimited. 

# Schema File Requirements
1) Each line of the file must specify the name of the class/attribute followed by the MySQL data-type of the class/attribute
2) The attributes must appear in the same order as they do in the data file
