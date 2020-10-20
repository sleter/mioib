#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<random>
#include <chrono>

std::random_device rd;
std::mt19937 gen(rd());

void print(std::vector<int> vec){
    for(auto i : vec){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

class Generator {
    std::vector<int> vec;
    int new_size = 0;

public:
    Generator(size_t n){
        vec = std::vector<int>(n);
        std::iota(vec.begin(), vec.end(), 0);
        new_size = n;
    }

    bool has_next(){
        return new_size > 0;
    }

    std::vector<int> next(){
        auto result = std::vector<int>(vec);

        std::uniform_int_distribution<int> dist(0,  new_size-1);
        std::iter_swap(vec.begin()+dist(gen), vec.begin()+new_size-1);

        --new_size;
        return result;
    }
};


int main(){

    long total_time = 0;
    int iters = 500;
    for(int i=0; i<iters; i++){
        auto start_time = std::chrono::high_resolution_clock::now();
        auto g = Generator(5000);
        while (g.has_next()){
            g.next();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    }

    // 5.346 ms for vector of 5000 and 500 iters 
    std::cout << (double)total_time/iters << " ms" << std::endl;
    return 0;
}