#ifndef OPTIMIZER_TABU_HPP
#define OPTIMIZER_TABU_HPP

#include "optimizer.hpp"
#include "randomutil.hpp"

optimizer_f tabu_optimizer(size_t initial_cadence, size_t elitar_size, size_t max_no_change_iterations);

#endif