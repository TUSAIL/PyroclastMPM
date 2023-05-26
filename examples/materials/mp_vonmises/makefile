DATEHASH=$(shell date +%Y%m%d_%H%M%S)

all: clean run postprocess

clean:
	rm -rf output/ plots

run: 
	mkdir -p output/uniaxial/
	mkdir -p output/simpleshear/
	python run.py

postprocess: 
	mkdir -p plots/uniaxial/
	mkdir -p plots/simpleshear/
	python postprocess.py


