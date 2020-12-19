#ifndef COSTS_HPP
#define COSTS_HPP

#include "common.hpp"

typedef std::pair<int, int> coords;

uint32_t euclidean_distance(const coords &from, const coords &to);

// Uses return value optimization (RVO), no copy
std::vector<coords> parse_file(const std::string &path);

struct cost_matrix
{
    std::vector<std::vector<uint32_t>> mat;
    size_t problem_size;

    cost_matrix(const std::vector<coords> &v);

    uint32_t &operator[](std::pair<size_t, size_t> &&pair);
    const uint32_t &operator[](std::pair<size_t, size_t> &&pair) const;
    uint32_t compute_cost(path_t &v) const;
    uint32_t evaluate_possible_cost(path_t &v, const uint32_t cost_before, size_t x, size_t y) const;
    
};

#endif