#ifndef OPTIMIZER_SA_HPP
#define OPTIMIZER_SA_HPP

#include "optimizer.hpp"
#include "randomutil.hpp"

optimizer_f simulated_anneling_optimizer(const float p, const float l_ratio, size_t max_no_change_iterations, float alpha);

#endif