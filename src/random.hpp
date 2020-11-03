#pragma once

#include<cstddef>
#include<vector>
#include<random>

/** Returns a random index from range <from, to> */
size_t random_index(size_t from, size_t to);

/** Shuffle vector with uniform distribution */
template <typename T>
void shuffle(std::vector<T> &v)
{
    for (size_t i = v.size() - 1; i > 0; --i)
    {
        std::swap(v[random_index(0, i)], v[i]);
    }
}

/** Returns random vector of integers from range <0, size-1> */
std::vector<int> random_vector(size_t size);