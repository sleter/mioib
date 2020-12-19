#ifndef OPTIMIZER_LOCAL_HPP
#define OPTIMIZER_LOCAL_HPP

#include "optimizer.hpp"
#include "randomutil.hpp"

/** Returns the best cost from all possible neighbours and permutated vector v as referrence (if better solution was found)*/
optimization_step steepest_optimizer_step(const cost_matrix &mat, path_t &v, const uint32_t cost);

/** Returns the first best cost and permutated vector v as referrence (if better solution was found)*/
optimization_step greedy_optimizer_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost);

optimizer_f local_search_optimizer(const optimizer_step_f step);

#endif