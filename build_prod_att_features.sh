#!/bin/bash

eval "$(conda shell.bash hook)"
. "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"

conda activate py3_5

conda info --env

function start_time() {
	echo ""
	echo "As of" `date` "generating product features for build set $build_set..."
}

function run_build() {
	python3.5 -u spacy.cluster.pipeline.prod_att.py $build_set > ./log/log.spacy.prod_att.cluster.pipeline.$build_set.log 2> ./log/err.spacy.prod_att.cluster.pipeline.$build_set.err
}

function count_features() {
	echo "Completed, as of "`date`", producing `cat $prod_features_file | wc -l` features in total."
}	

function error_exit() {
	echo "!!!! ERROR processing build set $build_set"
	exit 1
}

for build_set in {1..13}
do

	prod_features_file="data/product_attribute_features.$build_set.csv"
	start_time
	run_build
	test -s $prod_features_file && count_features || error_exit

done
