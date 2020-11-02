#!/bin/bash

rm -rf tsplib
mkdir -p tsplib
mkdir -p output

cd tsplib

echo $(pwd)
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz
tar -xzf ALL_tsp.tar.gz
rm ALL_tsp.tar.gz xray.problems.gz

gzip -d *.gz

for type in $(grep EDGE_WEIGHT_TYPE *.tsp | awk '{split($0, a, ":"); print(a[3])}' | sort -u)
do
    mkdir -p $type
    for filename in $(grep -l *.tsp -e $type)
    do
        tour=${filename%.*}.opt.tour
        if [ -f $tour ]; then
            awk -v DIR="$type" -v F="$tour" 'BEGIN {system("mv "F" "DIR"/")}'
        fi
    done 
    grep -l *.tsp -e $type | awk -v DIR=$type '{system("mv "$0" "DIR"/")}'
done

# mkdir -p $1
# grep -l *.tsp -e $1 | awk -v DIR=$1 '{system("mv "$0" "DIR"/")}'
