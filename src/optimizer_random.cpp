#include "optimizer_random.hpp"

optimizer_f time_constrained_optimizer(long limit_ms, const optimizer_step_f step)
{
    return [limit_ms, step](const cost_matrix &mat, path_t &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost);

        long elapsed = 0;
        auto start_time = now();
        do
        {
            auto step_result = step(mat, v, cost);
            ++result.seen_solutions;
            ++result.steps;
            if (step_result.cost < result.final_cost)
            {
                result.final_cost = step_result.cost;
            }
            elapsed = as_milliseconds(now() - start_time);
        } while (elapsed < limit_ms);

        result.final_path = std::vector<int>(v);
        return result;
    };
}

optimization_step random_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost)
{
    shuffle(v);
    return optimization_step(mat.compute_cost(v), 1);
}

optimization_step random_walk_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost)
{
    auto neighbour = random_neighbour(v);
    return optimization_step(mat.evaluate_possible_cost(v, prev_cost, neighbour.first, neighbour.second), 1);
}