#pragma once

#include <iostream>
#include "parser.hpp"

class cost_matrix
{
    std::vector<long> matrix;
    const size_t _size;

public:
    cost_matrix(const std::vector<coords> &v);

    const size_t size() const;

    const long at(const size_t row, const size_t col) const;

    const long compute_cost(std::vector<int> &v) const;

    const long evaluate_possible_cost(std::vector<int> &v, long cost_before, size_t x, size_t y) const;

    void print_matrix();
};