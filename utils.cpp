#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<random>
#include<chrono>
#include<functional>

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

    std::vector<int>& next(){
        if(new_size > 0 && new_size < vec.size()-1){
            std::swap(vec[random_index(0, new_size-1)], vec[new_size]);
        }
        --new_size;
        return vec;
    }
};

std::chrono::high_resolution_clock::time_point now(){
    return std::chrono::high_resolution_clock::now();
}

int64_t as_milliseconds(std::chrono::nanoseconds time){
    return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
}

int64_t measure_time(std::function<void()> f, long limit_ms){
    auto start_time = now();
    uint32_t iteration = 0;
    do
    {
        f();
        ++iteration;
        // std::cout << as_milliseconds(now() - start_time) << std::endl;
    } while (as_milliseconds(now() - start_time) < limit_ms);
    return as_milliseconds(now() - start_time) / (int64_t)iteration;
}

void test_function(){
    Generator g(5000);
    while (g.has_next()){
        g.next();
    }
}

int main(){

    // auto g = Generator(5);
    // while (g.has_next())
    // {
    //     print(g.next());
    // }
    
    long total_time = 0;
    int iters = 100;
    for(int i=0; i<iters; i++){
        total_time += measure_time(test_function, 100);
    }
    // 5.048 ms for vector of 5000 and 500 iters 
    std::cout << (double)total_time/iters << " ms" << std::endl;
    return 0;
}