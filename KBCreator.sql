use NEMO_KB;

drop table if exists ModelRepository;
drop table if exists AlgorithmResults;
create table AlgorithmResults(
	algorithm_id varchar(16),
	algorithm_name varchar(255), 
	accuracy double, 
	prec double, 
	recall double, 
	f1 double, 
	confusion_matrix varchar(255));


create table ModelRepository(
	algorithm_id varchar(16),
	algorithm_name varchar(255),
	args varchar(1000) -- this will store some aspect of the model --
);
