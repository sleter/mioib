#ifndef RESULTS_HPP
#define RESULTS_HPP

#include "common.hpp"

struct optimization_step
{
    size_t seen_solutions = 0;
    uint32_t cost = 0;

    optimization_step &update_cost(uint32_t new_cost);
    optimization_step(uint32_t cost = 0, size_t seen_solutions = 0);
};

struct optimization_header
{
    std::string problem;
    size_t problem_size = 0;

    std::string optimizer;
    size_t time_iterations = 0;
    long time_duration_ms = 0;
    uint32_t optimal_cost = 0;
    path_t optimal_path;

    optimization_header &with_names(const std::string &problem, const size_t problem_size, const std::string &optimizer, const uint32_t optimal_cost, const path_t &optimal_path);
    optimization_header(const std::string &problem, const uint32_t problem_size, const std::string &optimizer, const uint32_t optimal_cost, const path_t &optimal_path);
    optimization_header(const std::string &problem, const std::string &optimizer);
    optimization_header(size_t iterations, long duration);
    long time_ms() const;
    std::string to_csv() const;

    static const std::string csv_header;
};

struct optimization_result
{
    uint32_t start_cost = 0;
    uint32_t final_cost = 0;

    path_t final_path;

    size_t steps = 0;
    size_t seen_solutions = 0;

    optimization_result(const uint32_t cost);
    optimization_result(const uint32_t cost, path_t final_path);
    optimization_result();
    std::string to_csv() const;
    void best_solution(uint32_t cost, path_t &v);
    static const std::string csv_header;
};

struct optimization_raport
{
    optimization_header header;
    optimization_result result;

    optimization_raport(const optimization_header header, const optimization_result result);
    optimization_raport(const optimization_header header);
    friend std::ostream &operator<<(std::ostream &ostream, const optimization_raport &r);
    static const std::string csv_header;
};

#endif