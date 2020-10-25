Run utils.cpp using:
```
g++ utils.cpp -O3 && ./a.out
```

Use: `./download.sh` for getting data. Documentation for the format can be found [here](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf).

Nice formatting of cost matrix:
```
g++ -O3 utils.cpp && ./a.out tsplib/EUC_2D/eil51.tsp | column -t -s ' ' > output.txt
```