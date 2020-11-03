#!/bin/bash

for file in $(ls problems/*.tsp)
do
    echo "Processing file: "$file 
    ./utils $file 10 1000 0
done