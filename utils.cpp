#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<random>
#include <chrono>

std::random_device rd;
std::mt19937 gen(rd());

size_t random_index(size_t from, size_t to){
    std::uniform_int_distribution<size_t> distribution(from,  to);
    return distribution(gen);
}

void print(std::vector<int> vec){
    for(auto i : vec){ std::cout << i << " "; }
    std::cout << std::endl;
}

class Generator {
    std::vector<int> vec;
    int new_size = 0;

public:
    Generator(size_t n){
        vec = std::vector<int>(n);
        reset();
    }

    void reset(){
        std::iota(vec.begin(), vec.end(), 0);
        new_size = vec.size() - 1;
    }

    bool has_next(){
        return new_size >= 0;
    }

    std::vector<int> next(){
        if(new_size > 0 && new_size < vec.size()-1){
            std::swap(vec[random_index(0, new_size-1)], vec[new_size]);
        }
        --new_size;
        return vec;
    }
};


int main(){

    // auto g = Generator(5);
    // while (g.has_next())
    // {
    //     print(g.next());
    // }
    
    Generator g(5000);
    long total_time = 0;
    int iters = 500;
    for(int i=0; i<iters; i++){
        g.reset();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (g.has_next()){
            g.next();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    }

    // 5.048 ms for vector of 5000 and 500 iters 
    std::cout << (double)total_time/iters << " ms" << std::endl;
    return 0;
}