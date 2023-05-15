backupdir=$(shell date +%Y%m%d_%H%M%S)

all: clean run postprocess

backup:

	mkdir -p backups/${backupdir}
	cp -r output backups/${backupdir}/output
	cp -r plots backups/${backupdir}/plots
	cp -r logs backups/${backupdir}/logs
	cp -r *.py backups/${backupdir}/
	cp -r *.toml backups/${backupdir}/
	
clean: 
	mkdir -p output plots logs
	rm -rf output/* plots/* logs/*

clean_backup:
	rm -rf backups/*

run:
	date >> logs/logfile.txt
	python run.py >>  logs/logfile.txt

run_time:
	date >> logs/logfile.txt
	(time python run.py ) >>  logs/logfile.txt  2>  logs/runtime.txt 

postprocess:
	python postprocess.py >> logs/logfile.txt