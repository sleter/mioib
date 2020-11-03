#include "costmatrix.hpp"

cost_matrix::cost_matrix(const std::vector<coords> &v) : matrix(v.size() * v.size()), _size(v.size())
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        for (size_t j = i + 1; j < v.size(); ++j)
        {
            long distance = euclidean_distance(v[i], v[j]);
            matrix[i * _size + j] = distance;
            matrix[j * _size + i] = distance;
        }
    }
}

const size_t cost_matrix::size() const {
    return _size;
}

const long cost_matrix::at(const size_t row, const size_t col) const
{
    return matrix[row * _size + col];
}

const long cost_matrix::compute_cost(std::vector<int> &v) const
{
    long cost = matrix[v[0]*_size + v[v.size() - 1]];
    for (size_t i = 1; i < v.size(); ++i)
    {
        cost += matrix[v[i - 1]*_size + v[i]];
    }
    return cost;
}

const long cost_matrix::evaluate_possible_cost(std::vector<int> &v, long cost_before, size_t x, size_t y) const
{
    size_t x_prev = x - 1;
    if (x == 0) // Fallback to the vector end
        x_prev = v.size() - 1;

    size_t y_succ = y + 1;
    if (y_succ >= v.size())
        y_succ = 0;

    const long removed_cost = matrix[v[x_prev]*_size + v[x]] + matrix[v[y]*_size + v[y_succ]];
    const long new_cost = matrix[v[x_prev]*_size + v[y]] + matrix[v[x]*_size + v[y_succ]];

    if (x_prev == y && x == y_succ)
        return cost_before;
    else
        return cost_before - removed_cost + new_cost;
}

void cost_matrix::print_matrix()
{
    size_t printed = 0;
    for (auto &value : matrix)
    {
        std::cout << value << ' ';
        ++printed;
        if (printed == _size)
        {
            printed = 0;
            std::cout << '\n';
        }
    }
}
