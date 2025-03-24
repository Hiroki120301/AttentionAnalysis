#!/bin/bash

DATASET="2wikimqa"

# run evaluation
python main.py --model LLaMA-2-7B-32K --task $DATASET

# check accuracy
# python eval.py --model LLaMA-2-7B-32K --use_centroids --percentile $PERCENTILE --percent_clusters $PERC_CLUSTERS
