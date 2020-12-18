#include "optimizer_local.hpp"

/** Returns the best cost from all possible neighbours and permutated vector v as referrence (if better solution was found)*/
optimization_step steepest_optimizer_step(const cost_matrix &mat, path_t &v, const uint32_t cost)
{
    bool found = false;
    size_t best_from, best_to;

    auto result = optimization_step(cost);
    uint32_t best_cost = cost;

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            uint32_t next_cost = mat.evaluate_possible_cost(v, cost, from, to);
            ++result.seen_solutions;

            if (next_cost < best_cost)
            {
                found = true;
                best_cost = next_cost;
                best_from = from;
                best_to = to;
            }
        }
    }

    if (found)
    {
        swap_with_rotation(v, best_from, best_to);
    }

    return result.update_cost(best_cost);
}

/** Returns the first best cost and permutated vector v as referrence (if better solution was found)*/
optimization_step greedy_optimizer_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost)
{
    auto result = optimization_step(prev_cost);

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            uint32_t next_cost = mat.evaluate_possible_cost(v, prev_cost, from, to);
            ++result.seen_solutions;
            if (next_cost < prev_cost)
            {
                swap_with_rotation(v, from, to);
                return result.update_cost(next_cost);
            }
        }
    }

    return result.update_cost(prev_cost);
}

optimizer_f local_search_optimizer(const optimizer_step_f step)
{
    return [step](const cost_matrix &mat, path_t &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost);

        while (true)
        {
            auto step_result = step(mat, v, result.final_cost);
            uint32_t new_cost = step_result.cost;
            result.seen_solutions += step_result.seen_solutions;
            ++result.steps;
            if (new_cost < result.final_cost)
            {
                result.final_cost = new_cost;
            }
            else
            {
                result.final_path = std::vector<int>(v);
                return result;
            }
        }
    };
}