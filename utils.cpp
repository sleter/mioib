#include<iostream>
#include<vector>
#include<numeric>

void print(std::vector<int> vec){
    for(auto i : vec){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

class Generator {
    std::vector<int> vec;
    int skip = 0;

public:
    Generator(size_t n){
        vec = std::vector<int>(n);
        std::iota(vec.begin(), vec.end(), 0);
    }

    bool has_next(){
        return skip < vec.size();
    }

    std::vector<int> next(){
        auto result = std::vector<int>(vec);
        ++skip;
        vec[0] = 100;
        return result;
    }
};


int main(){
    auto g = Generator(3);

    while(g.has_next()){
        print(g.next());
    }

    return 0;
}