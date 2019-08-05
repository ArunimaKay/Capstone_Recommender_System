#!/bin/bash

eval "$(conda shell.bash hook)"
. "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"

conda activate py3.5

conda info --env

function start_time() {
	echo ""
	echo "As of" `date` "generating product review features for build set $build_set..."
}

function run_build() {
	python3.5 -u spacy.cluster.pipeline.prod_rev.py $build_set > ./log/log.spacy.prod_rev.cluster.pipeline.$build_set.log 2> ./log/err.spacy.prod_rev.cluster.pipeline.$build_set.err
}

function dedupe_features() {
        cat $prod_features_file | sort -u > $dedupe_prod_features_file
}

function dedupe_tf_features() {
        cat $prod_features_tf_file | sort -u > $dedupe_prod_features_tf_file
}

function count_features() {
	echo "Completed, as of "`date`", producing `cat $dedupe_prod_features_file | wc -l` features in total."
}	

function error_exit() {
	echo "!!!! ERROR processing build set $build_set"
	exit 1
}

for build_set in {5..13}
do

	prod_features_file="data/product_review_features.$build_set.csv"
	dedupe_prod_features_file="data/sorted/product_review_features.$build_set.csv"

	prod_features_tf_file="data/product_review_tf_features.$build_set.csv"
        dedupe_prod_features_tf_file="data/sorted/product_review_tf_features.$build_set.csv"

	start_time
	run_build
	dedupe_features
	dedupe_tf_features
	test -s $prod_features_file && count_features || error_exit

done
