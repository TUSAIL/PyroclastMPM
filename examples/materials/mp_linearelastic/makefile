DATEHASH=$(shell date +%Y%m%d_%H%M%S)

all: clean run postprocess

clean:
	rm -rf output plots

run: run_uniaxial run_simpleshear

run_uniaxial:
	mkdir -p output/uniaxial/ 
	python run_uniaxial.py

run_simpleshear:
	mkdir -p output/simpleshear/ 
	python run_simpleshear.py

postprocess: postprocess_uniaxial postprocess_simpleshear

postprocess_uniaxial:
	mkdir -p plots/uniaxial/
	python postprocess_uniaxial.py


postprocess_simpleshear:
	mkdir -p plots/simpleshear/
	python postprocess_simpleshear.py
