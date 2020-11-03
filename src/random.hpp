#pragma once

#include <cstddef>
#include <vector>
#include <random>
#include <chrono>

class random_utils
{
    std::default_random_engine engine;
    std::mt19937 generator;

public:
    random_utils();

    /** Returns a random index from range <from, to> */
    const size_t index(const size_t from, const size_t to);

    /** Shuffle vector with uniform distribution */
    void shuffle(std::vector<int> &v);

    /** Returns random vector of integers from range <0, size-1> */
    std::vector<int> vector(const size_t size);
};
