#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "common.hpp"
#include "results.hpp"
#include "costs.hpp"

using optimizer_f = std::function<optimization_result(const cost_matrix &, path_t &, const uint32_t)>;

#endif