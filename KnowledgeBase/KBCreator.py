import KnowledgeBase

def main():
	kb = KnowledgeBase.KnowledgeBase("config/config.json")
	stmts = [ 'drop table if exists ModelRepository','drop table if exists AlgorithmResults', 'drop table if exists CurrentModel', 'drop table if exists CurrentlyOptimizingModels', 'create table AlgorithmResults(algorithm_id varchar(16), algorithm_name varchar(255), data_source varchar(255), accuracy double, prec double, recall double, f1 double, confusion_matrix varchar(512))', 'create table ModelRepository( algorithm_id varchar(16), algorithm_name varchar(255), arg_type varchar(255), arg_val varchar(255) )', 'create table CurrentModel( algorithm_id varchar(16) primary key)', 'create table CurrentlyOptimizingModels( algorithm_id varchar(16) primary key)']

	for stmt in stmts:
 		kb.executeQuery(stmt)

if __name__ == '__main__':
	main()
