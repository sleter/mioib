#include "optimizer_heuristic.hpp"

optimization_result heuristic_optimizer(const cost_matrix &mat, path_t &v, const uint32_t cost)
{
    optimization_result result(cost);

    for (size_t i = 1; i < v.size(); ++i)
    {
        size_t prev_node = i - 1;
        size_t idx_to_swap = i;
        uint32_t prev_cost = mat[{v[i], v[prev_node]}];
        ++result.steps;

        for (size_t j = i + 1; j < v.size(); ++j)
        {
            ++result.seen_solutions;
            uint32_t new_cost = mat[{v[j], v[prev_node]}];
            if (new_cost < prev_cost)
            {
                prev_cost = new_cost;
                idx_to_swap = j;
            }
        }
        if (i != idx_to_swap)
        {
            std::swap(v[i], v[idx_to_swap]);
        }
    }

    result.final_path = std::vector<int>(v);
    result.final_cost = mat.compute_cost(v);
    return result;
}