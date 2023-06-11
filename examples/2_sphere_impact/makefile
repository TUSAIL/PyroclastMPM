DATEHASH=$(shell date +%Y%m%d_%H%M%S)

all: clean run postprocess

backup:
	mkdir -p backups/${dir}
	cp -r output backups/${dir}/output
	cp -r plots backups/${dir}/plots
	cp -r logs backups/${dir}/logs
	cp -r *.py backups/${dir}/
	cp -r *.toml backups/${dir}/

backup_date:
	mkdir -p backups/$(DATEHASH)
	cp -r output backups/${DATEHASH}/output
	cp -r plots backups/${DATEHASH}/plots
	cp -r logs backups/${DATEHASH}/logs
	cp -r *.py backups/${DATEHASH}/
	cp -r *.toml backups/${DATEHASH}/
	
clean: 
	mkdir -p output plots logs
	rm -rf output/* plots/* logs/*

run:
	date >> logs/logfile.txt
	python run.py 

postprocess:
	python postprocess.py >> logs/logfile.txt