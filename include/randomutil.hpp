#ifndef RANDOMUTIL_HPP
#define RANDOMUTIL_HPP

#include "common.hpp"
#include "costs.hpp"

size_t random_index(size_t from, size_t to);
float random_float();

/** Shuffle vector with uniform distribution */
void shuffle(path_t &v);
path_t random_vector(size_t size);

std::pair<size_t, size_t> random_neighbour(path_t &v);

std::tuple<uint32_t, size_t, size_t> random_neighbour(const cost_matrix &mat, path_t &v, const uint32_t cost);

float mean_neighbour_cost(const cost_matrix &mat, size_t samples);

#endif