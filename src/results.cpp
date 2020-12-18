#include "results.hpp"

optimization_step &optimization_step::update_cost(uint32_t new_cost)
{
    cost = new_cost;
    return *this;
}

optimization_step::optimization_step(uint32_t cost, size_t seen_solutions) : cost(cost), seen_solutions(seen_solutions) {}

optimization_header &optimization_header::with_names(const std::string &problem, const size_t problem_size, const std::string &optimizer, const uint32_t optimal_cost, const path_t &optimal_path)
{
    this->problem = problem;
    this->problem_size = problem_size;
    this->optimizer = optimizer;
    this->optimal_cost = optimal_cost;
    this->optimal_path = optimal_path;
    return *this;
}

optimization_header::optimization_header(const std::string &problem,
                                         const uint32_t problem_size,
                                         const std::string &optimizer,
                                         const uint32_t optimal_cost,
                                         const path_t &optimal_path) : problem(problem),
                                                                       optimizer(optimizer),
                                                                       time_iterations(0),
                                                                       problem_size(problem_size),
                                                                       optimal_cost(optimal_cost),
                                                                       optimal_path(optimal_path),
                                                                       time_duration_ms(0) {}

optimization_header::optimization_header(const std::string &problem,
                                         const std::string &optimizer) : problem(problem),
                                                                         optimizer(optimizer),
                                                                         time_iterations(0),
                                                                         time_duration_ms(0) {}

optimization_header::optimization_header(size_t iterations, long duration) : time_iterations(iterations), time_duration_ms(duration) {}

long optimization_header::time_ms() const
{
    if (time_iterations <= 0 || time_iterations <= 0)
        return 1;
    else
        return time_duration_ms / time_iterations;
}

std::string optimization_header::to_csv() const
{
    std::stringstream ss;
    ss << problem << ',' << problem_size << ',' << optimizer << ',' << time_iterations << ',' << time_duration_ms << ',' << time_ms() << ',' << optimal_cost << ',' << as_string(optimal_path);
    return ss.str();
}

optimization_result::optimization_result(const uint32_t cost) : start_cost(cost), final_cost(cost) {}
optimization_result::optimization_result(const uint32_t cost, path_t final_path) : start_cost(cost), final_cost(cost), final_path(final_path) {}
optimization_result::optimization_result() {}

std::string optimization_result::to_csv() const
{
    std::stringstream ss;
    ss << start_cost << ',' << final_cost << ',' << steps << ',' << seen_solutions << ',' << as_string(final_path);

    return ss.str();
}

void optimization_result::best_solution(uint32_t cost, path_t &v)
{
    final_cost = cost;
    final_path = v;
}

optimization_raport::optimization_raport(const optimization_header header, const optimization_result result) : header(header), result(result) {}

optimization_raport::optimization_raport(const optimization_header header) : header(header) {}

std::ostream &operator<<(std::ostream &ostream, const optimization_raport &r)
{
    ostream << r.header.to_csv() << ',' << r.result.to_csv() << '\n';
    return ostream;
}

const std::string optimization_header::csv_header = "problem,problem_size,optimizer,time_iterations,time_duration_ms,time_ms,optimal_cost,optimal_path";
const std::string optimization_result::csv_header = "start_cost,final_cost,steps,seen_solutions,final_path";
const std::string optimization_raport::csv_header = optimization_header::csv_header + ',' + optimization_result::csv_header;