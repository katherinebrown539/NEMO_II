use NEMO_KB;

drop table if exists AlgorithmResults;
create table AlgorithmResults(
	algorithm_id int,
	algorithm_name varchar(255), 
	accuracy double, 
	prec double, 
	recall double, 
	f1 double, 
	confusion_matrix varchar(255));

/*
drop table if exists ModelRepository;
create table ModelRepository(
	algorithm_id int primary key,
	algorithm_name varchar(255),
	args varchar(1000) --this stores all the arguments required to recreate the model. . .
);
*/