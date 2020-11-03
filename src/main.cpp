#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>
#include <fstream>
#include <string>
#include <regex>
#include <memory>
#include <math.h>
#include <map>
#include <list>
#include <set>

#include "random.hpp"
#include "parser.hpp"
#include "costmatrix.hpp"

template <typename T>
std::string as_string(const std::vector<T> &vec)
{
    std::ostringstream os;
    for (auto &i : vec)
    {
        os << i << " ";
    }

    std::string result = os.str();
    if (result.size() > 0)
    {
        result.pop_back();
    }
    return result;
}

template <typename T>
void print(const std::vector<T> &vec)
{
    std::cout << as_string(vec) << std::endl;
}

template <typename T>
void print(const std::list<T> &list)
{
    for (auto &i : list)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}


template <typename T>
void swap_with_rotation(std::vector<T> &v, size_t from, size_t to)
{
    while (to > from)
    {
        std::swap(v[from], v[to]);
        ++from;
        --to;
    }
}

/** Returns the best cost from all possible neighbours and permutated vector v as referrence (if better solution was found)*/
float inline steepest_optimizer_step(const cost_matrix &mat, std::vector<int> &v, const float cost)
{
    bool found = false;
    size_t best_from, best_to;
    float best_cost = cost;

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            float next_cost = mat.evaluate_possible_cost(v, cost, from, to);
            if (next_cost < best_cost)
            {
                found = true;
                best_cost = next_cost;
                best_from = from;
                best_to = to;
            }
        }
    }

    if (found)
    {
        swap_with_rotation(v, best_from, best_to);
    }

    return best_cost;
}

/** Returns the first best cost and permutated vector v as referrence (if better solution was found)*/
float inline greedy_optimizer_step(const cost_matrix &mat, std::vector<int> &v, const float prev_cost)
{
    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            float next_cost = mat.evaluate_possible_cost(v, prev_cost, from, to);
            if (next_cost < prev_cost)
            {
                swap_with_rotation(v, from, to);
                return next_cost;
            }
        }
    }

    return prev_cost;
}

auto local_search_optimizer(const std::function<float(const cost_matrix &mat, std::vector<int> &v, const float cost)> step)
{
    return [step](const cost_matrix &mat, std::vector<int> &v, const float cost) {
        float prev_cost = cost;
        while (true)
        {
            float new_cost = step(mat, v, prev_cost);
            if (new_cost < prev_cost)
            {
                prev_cost = new_cost;
            }
            else
            {
                return prev_cost;
            }
        }
    };
}

// auto random_walk_optimizer(const long limit_ms)
// {
//     return [limit_ms](const cost_matrix &mat, std::vector<int> &v, const float cost) {

//     };
// }

float random_optimizer(const cost_matrix &mat, std::vector<int> &v, const float cost){
    shuffle(v);
    return mat.compute_cost(v);
}

float heuristic_optimizer(const cost_matrix &mat, std::vector<int> &v, const float cost)
{
    for (size_t i = 1; i < v.size(); ++i)
    {
        size_t prev_node = i - 1;
        size_t idx_to_swap = i;
        float prev_cost = mat.at(v[i], v[prev_node]);

        for (size_t j = i + 1; j < v.size(); ++j)
        {
            float new_cost = mat.at(v[j], v[prev_node]);
            if (new_cost < prev_cost)
            {
                prev_cost = new_cost;
                idx_to_swap = j;
            }
        }
        if (i != idx_to_swap)
        {
            std::swap(v[i], v[idx_to_swap]);
        }
    }

    return mat.compute_cost(v);
}

std::chrono::high_resolution_clock::time_point now()
{
    return std::chrono::high_resolution_clock::now();
}

int64_t as_milliseconds(std::chrono::nanoseconds time)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
}

struct tsp_optimizer
{
    const std::string name;
    const bool shuffle;
    const std::function<float(const cost_matrix &, std::vector<int> &, const float)> optimizer;

    tsp_optimizer(std::string name, bool shuffle, std::function<float(const cost_matrix &, std::vector<int> &, const float)> optimizer) : name(name), shuffle(shuffle), optimizer(optimizer) {}

    const inline float operator()(const cost_matrix &mat, std::vector<int> &v, const float prev_cost) const
    {
        return optimizer(mat, v, prev_cost);
    }
};

class tsp
{
    std::list<std::string> optimizer_names;
    std::list<size_t> loops;
    std::list<long> elapsed_time_ms;
    std::list<float> best_costs;
    std::list<float> last_costs;
    std::list<std::vector<int>> best_paths;

    const cost_matrix mat;
    const size_t iterations;
    const long limit_ms;
    const bool print_path;

    void measure_time(const tsp_optimizer &optimizer, std::vector<int> &v)
    {
        long elapsed = 0;
        uint32_t iteration = 0;
        float best_cost = MAXFLOAT;
        float last_cost = MAXFLOAT;

        auto start_time = now();

        do
        {
            float local_cost = mat.compute_cost(v);
            float next_cost = optimizer(mat, v, local_cost);
            last_cost = next_cost;

            if (next_cost < best_cost)
            {
                best_cost = next_cost;
            }

            ++iteration;
            elapsed = as_milliseconds(now() - start_time);
        } while (elapsed < limit_ms);

        elapsed_time_ms.emplace_back(elapsed);
        loops.emplace_back(iteration);
        best_costs.emplace_back(best_cost);
        last_costs.emplace_back(last_cost);
        optimizer_names.emplace_back(optimizer.name);
        best_paths.push_back(v);
    }

public:
    tsp(const cost_matrix &mat, size_t iterations, long limit_ms, bool print_path) : mat(mat), iterations(iterations), limit_ms(limit_ms), print_path(print_path) {}

    void run(const tsp_optimizer &optimizer)
    {
        for (size_t idx = 0; idx < iterations; ++idx)
        {
            std::vector<int> v(mat.size());
            std::iota(v.begin(), v.end(), 0);

            if(optimizer.shuffle) shuffle(v);
            measure_time(optimizer, v);
        }
    }

    void add_optimal_tour(std::pair<float, std::vector<int>> &optimal)
    {
        elapsed_time_ms.emplace_back(0);
        loops.emplace_back(0);
        best_costs.emplace_back(optimal.first);
        last_costs.emplace_back(optimal.first);
        optimizer_names.emplace_back("optimal");
        best_paths.push_back(optimal.second);
    }

    friend std::ostream &operator<<(std::ostream &ostream, const tsp &t)
    {
        auto name_it = t.optimizer_names.begin();
        auto cost_it = t.best_costs.begin();
        auto last_cost_it = t.last_costs.begin();
        auto time_it = t.elapsed_time_ms.begin();
        auto loop_it = t.loops.begin();
        auto path_it = t.best_paths.begin();

        ostream << "optimizer_name,best_cost,last_cost,elapsed_time_ms,loop_count";
        if (t.print_path)
        {
            ostream << ",best_path";
        }
        ostream << '\n';

        while (name_it != t.optimizer_names.end() && cost_it != t.best_costs.end() && last_cost_it != t.last_costs.end() && time_it != t.elapsed_time_ms.end() && loop_it != t.loops.end() && path_it != t.best_paths.end())
        {
            ostream << *name_it << ',' << *cost_it << ',' << *last_cost_it << ',' << *time_it << ',' << *loop_it;
            if (t.print_path)
            {
                ostream << ',' << as_string(*path_it);
            }
            ostream << '\n';

            ++name_it;
            ++cost_it;
            ++last_cost_it;
            ++time_it;
            ++loop_it;
            ++path_it;
        }

        return ostream;
    }
};

const std::string file_name(const std::string path)
{
    std::regex re("(.*/)?(\\w+)\\.tsp");
    std::smatch matches;
    std::regex_match(path, matches, re);
    return std::string(matches[matches.size() - 1]);
}

const std::string output_path(const std::string intput_path, const long limit_ms)
{
    return "output/" + file_name(intput_path) + "_" + std::to_string(limit_ms) + ".csv";
}

const std::pair<float, std::vector<int>> optimal_tour(const cost_matrix &mat, std::string path)
{
    std::string opt_tour = ".opt.tour";
    std::string optimal_solution_path = path;
    optimal_solution_path.replace(optimal_solution_path.find(".tsp"), opt_tour.length(), opt_tour);

    std::vector<int> vec;
    std::ifstream file(optimal_solution_path);
    if (!file.good())
        return std::make_pair(-1, vec);

    if (file.is_open())
    {
        std::string line;
        int node;

        std::regex re("^(\\w+)\\W*:\\W*(\\w+)$");
        std::smatch matches;

        while (std::getline(file, line))
        {
            // Reserve needed memory
            if (std::regex_match(line, matches, re) && matches[1].compare("DIMENSION") == 0)
                vec.reserve(std::stoi(matches[2]));

            else if (line.compare("TOUR_SECTION") == 0)
                break;
        }

        while (file >> node)
        {
            if (node < 0)
                break;
            vec.emplace_back(node - 1);
        }

        file.close();

        float cost = mat.compute_cost(vec);
        return std::make_pair(cost, vec);
    }
    else
    {
        return std::make_pair(-1, vec);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << " <file>.tsp <iterations> <time> <print_path>" << '\n';
        return 0;
    }

    std::string path(argv[1]);
    size_t n = std::stoul(argv[2]);
    long limit_ms = std::stol(argv[3]);
    bool print_path = std::stoi(argv[4]) > 0;

    std::list<tsp_optimizer> optimizers = {
        tsp_optimizer("steepest", true, local_search_optimizer(steepest_optimizer_step)),
        tsp_optimizer("greedy", true, local_search_optimizer(greedy_optimizer_step)),
        tsp_optimizer("heuristic", true, heuristic_optimizer),
        tsp_optimizer("random", false, random_optimizer)};

    auto coords = parse_file(path);
    const auto mat = cost_matrix(coords);
    auto v = random_vector(mat.size());

    auto optimal = optimal_tour(mat, path);
    if (optimal.first < 0)
    {
        std::cout << "Cannot find optimal file for TSP " << path << '\n';
        return -1;
    }

    auto problem = tsp(mat, n, limit_ms, print_path);
    problem.add_optimal_tour(optimal);

    for (auto &optimizer : optimizers)
    {
        std::cout << "Running optimizer: " << optimizer.name << '\n';
        problem.run(optimizer);
    }

    std::ofstream file;
    file.open(output_path(path, limit_ms));
    file << problem;
    file.close();

    return 0;
}