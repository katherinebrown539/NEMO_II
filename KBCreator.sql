use NEMO_KB;

drop table if exists AlgorithmResults;
create table AlgorithmResults(
	algorithm_id varchar(48),
	algorithm_name varchar(255), 
	accuracy double, 
	prec double, 
	recall double, 
	f1 double, 
	confusion_matrix varchar(255));
