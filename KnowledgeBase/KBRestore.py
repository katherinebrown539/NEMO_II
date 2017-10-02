import KnowledgeBase

def main():
	kb = KnowledgeBase.KnowledgeBase("config/config.json")
	stmts = ['drop table if exists CurrentlyOptimizingModels', 'create table CurrentlyOptimizingModels(algorithm_id varchar(16) primary key)']
	for stmt in stmts:
		kb.executeQuery(stmt)

if __name__ == '__main__':
	main()
