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

void shuffle(std::vector<int>& v){
    for(size_t i = v.size() - 1; i > 0; --i){
        std::swap(v[random_index(0, i)], v[i]);
    }
}

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
    std::vector<int> v(1000000);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v);
}

int main(){
    long total_time = 0;
    int iters = 10;
    for(int i=0; i<iters; i++){
        total_time += measure_time(test_function, 100);
    }
    // 5.048 ms for vector of 5000 and 500 iters 
    std::cout << (double)total_time/iters << " ms" << std::endl;
    return 0;
}