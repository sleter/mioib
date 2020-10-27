#!/bin/bash

mkdir -p tsplib
cd tsplib

echo $(pwd)
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz
tar -xzf ALL_tsp.tar.gz
rm ALL_tsp.tar.gz xray.problems.gz *.tour.gz

gzip -d *.gz

for type in $(grep EDGE_WEIGHT_TYPE * | awk '{split($0, a, ":"); print(a[3])}' | sort -u)
do
    mkdir -p $type
    grep -l *.tsp -e $type | awk -v DIR=$type '{system("mv "$0" "DIR"/")}'
done

# mkdir -p $1
# grep -l *.tsp -e $1 | awk -v DIR=$1 '{system("mv "$0" "DIR"/")}'
