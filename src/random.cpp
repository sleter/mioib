#include "random.hpp"

std::default_random_engine rd{static_cast<long unsigned int>(time(0))};
std::mt19937 gen(rd());

size_t random_index(size_t from, size_t to)
{
    std::uniform_int_distribution<size_t> distribution(from, to);
    return distribution(gen);
}

std::vector<int> random_vector(size_t size)
{
    std::vector<int> v(size);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v);
    return v;
}