#ifndef OPTIMIZER_HEURISTIC_HPP
#define OPTIMIZER_HEURISTIC_HPP

#include "optimizer.hpp"
#include "randomutil.hpp"

optimization_result heuristic_optimizer(const cost_matrix &mat, path_t &v, const uint32_t cost);

#endif