#include "optimizer_sa.hpp"

optimizer_f simulated_anneling_optimizer(const float p, const float l_ratio, size_t max_no_change_iterations, float alpha)
{
    return [p, l_ratio, max_no_change_iterations, alpha](const cost_matrix &mat, path_t &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost);

        float mean_next_cost = mean_neighbour_cost(mat, 10000);
        const float c0 = (-(abs(mean_next_cost - cost))) / log(p);

        uint32_t neighbourhood_size = (v.size() * (v.size() - 1)) / 2;
        uint32_t L = static_cast<uint32_t>(round(neighbourhood_size * l_ratio));

        uint32_t current_cost = cost;
        result.final_cost = cost;

        float temperature = c0;
        size_t no_improvement_count = 0;

        while (true)
        {
            ++result.steps;
            uint32_t step_start_cost = result.final_cost;
            for (size_t j = 0; j < L; ++j)
            {
                auto next_solution = random_neighbour(mat, v, current_cost);
                ++result.seen_solutions;
                uint32_t next_cost = std::get<0>(next_solution);
                size_t from = std::get<1>(next_solution);
                size_t to = std::get<2>(next_solution);

                if (next_cost <= current_cost)
                {
                    swap_with_rotation(v, from, to);
                    current_cost = next_cost;
                    if (current_cost < result.final_cost)
                        result.best_solution(current_cost, v);
                }
                else
                {
                    float probability = exp(-((float)next_cost - (float)current_cost) / temperature);
                    if (probability > random_float())
                    {
                        swap_with_rotation(v, from, to);
                        current_cost = next_cost;
                        if (current_cost < result.final_cost)
                            result.best_solution(current_cost, v);
                    }
                }
            }

            temperature *= alpha;
            if (current_cost < step_start_cost)
            {
                no_improvement_count = 0;
            }
            else
            {
                ++no_improvement_count;
            }

            if (no_improvement_count >= max_no_change_iterations && temperature < 0.01)
            {
                break;
            }
        }

        return result;
    };
}