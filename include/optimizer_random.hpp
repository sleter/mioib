#ifndef OPTIMIZER_RANDOM_HPP
#define OPTIMIZER_RANDOM_HPP

#include "optimizer.hpp"
#include "randomutil.hpp"

optimizer_f time_constrained_optimizer(long limit_ms, const optimizer_step_f step);

optimization_step random_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost);

optimization_step random_walk_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost);

#endif