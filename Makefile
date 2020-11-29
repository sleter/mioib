all:
	g++ -O3 -std=c++17 -lstdc++fs -c src/main.cpp
	mkdir -p bin
	mv *.o bin/
	g++ -o main bin/*.o

clean:
	rm -f utils
