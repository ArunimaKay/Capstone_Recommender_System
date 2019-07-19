#!/bin/bash

conda activate py3.5

function start_time() {
	echo ""
	echo "As of" `date` "generating product features for build set $build_set..."
}

function run_build() {
	python3.5 -u spacy.cluster.pipeline.py $build_set > log.spacy.cluster.pipeline.$build_set.log 2> err.spacy.cluster.pipeline.$build_set.err
}

function count_features() {
	echo "Completerd, as of "`date`", producing `cat $prod_features_file | wc -l` features in total."
}	

function error_exit() {
	echo "!!!! ERROR processing build set $build_set"
	exit 1
}

for build_set in {3..7}
do

	prod_features_file="data/product_features.$build_set.csv"
	start_time
	run_build
	test -s $prod_features_file && count_features || error_exit

done
