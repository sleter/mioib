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
#include <math.h>
#include <list>

std::default_random_engine rd{static_cast<long unsigned int>(time(0))};
std::mt19937 gen(rd());

std::chrono::high_resolution_clock::time_point now()
{
    return std::chrono::high_resolution_clock::now();
}

int64_t as_milliseconds(std::chrono::nanoseconds time)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
}

size_t random_index(size_t from, size_t to)
{
    std::uniform_int_distribution<size_t> distribution(from, to);
    return distribution(gen);
}

template <typename T>
std::string as_string(const std::vector<T> &vec)
{
    std::ostringstream os;
    for (auto &i : vec)
        os << i << " ";

    std::string result = os.str();

    if (result.size() > 0)
        result.pop_back();

    return result;
}

struct optimization_header
{
    std::string problem;
    std::string optimizer;
    size_t time_iterations = 0;
    long time_duration_ms = 0;

    optimization_header &with_names(const std::string &problem, const std::string &optimizer)
    {
        this->problem = problem;
        this->optimizer = optimizer;
        return *this;
    }

    optimization_header(const std::string &problem, const std::string &optimizer) : problem(problem), optimizer(optimizer), time_iterations(0), time_duration_ms(0) {}

    optimization_header(size_t iterations, long duration) : time_iterations(iterations), time_duration_ms(duration) {}

    long time_ms() const
    {
        if (time_iterations <= 0 || time_iterations <= 0)
            return 1;
        else
            return time_duration_ms / time_iterations;
    }

    std::string to_csv() const
    {
        std::stringstream ss;
        ss << problem << ',' << optimizer << ',' << time_iterations << ',' << time_duration_ms << ',' << time_ms();
        return ss.str();
    }

    static const std::string csv_header;
};

const std::string optimization_header::csv_header = "problem,optimizer,time_iterations,time_duration_ms,time_ms";

struct optimization_step
{
    size_t seen_solutions = 0;
    uint32_t score = 0;

    optimization_step &update_score(uint32_t new_score)
    {
        score = new_score;
        return *this;
    }

    optimization_step(uint32_t score, size_t seen_solutions = 0) : score(score), seen_solutions(seen_solutions) {}
};

struct optimization_result
{
    uint32_t start_score = 0;
    uint32_t final_score = 0;

    size_t steps = 0;
    size_t seen_solutions = 0;

    std::vector<int> start_path;
    std::vector<int> final_path;

    optimization_result(const std::pair<uint32_t, std::vector<int>> &optimal) : start_score(optimal.first), final_score(optimal.first),
                                                                             start_path(optimal.second), final_path(optimal.second) {}

    optimization_result(const uint32_t score, const std::vector<int> &start_path) : start_score(score), final_score(score), start_path(start_path) {}

    optimization_result() {}

    std::string to_csv() const
    {
        std::stringstream ss;
        ss << start_score << ',' << final_score << ',' << steps << ',' << seen_solutions << ','
           << as_string(start_path) << ',' << as_string(final_path);

        return ss.str();
    }

    static const std::string csv_header;
};

const std::string optimization_result::csv_header = "start_score,final_score,steps,seen_solutions,start_path,final_path";

struct optimization_raport
{
    optimization_header header;
    optimization_result result;

    optimization_raport(const optimization_header header, const optimization_result result) : header(header), result(result) {}

    optimization_raport(const optimization_header header) : header(header) {}

    friend std::ostream &operator<<(std::ostream &ostream, const optimization_raport &r)
    {
        ostream << r.header.to_csv() << ',' << r.result.to_csv() << '\n';
        return ostream;
    }

    static const std::string csv_header;
};

const std::string optimization_raport::csv_header = optimization_header::csv_header + ',' + optimization_result::csv_header;

typedef std::pair<int,int> coords;

uint32_t euclidean_distance(const coords &from, const coords &to)
{
    int64_t dx = from.first - to.first;
    int64_t dy = from.second - to.second;

    int64_t distance = dx * dx + dy * dy;
    return static_cast<uint32_t>(round(sqrt(distance)));
}

// Uses return value optimization (RVO), no copy
std::vector<coords> parse_file(const std::string &path)
{
    auto vec = std::vector<coords>();

    std::ifstream file;
    std::string line;

    file.open(path);

    if (file.is_open())
    {
        int nodeId;
        uint32_t x, y;

        std::regex re("^(\\w+)\\W*:\\W*(\\w+)$");
        std::smatch matches;

        while (std::getline(file, line))
        {
            // Reserve needed memory
            if (std::regex_match(line, matches, re) && matches[1].compare("DIMENSION") == 0)
                vec.reserve(std::stoi(matches[2]));

            else if (line.compare("NODE_COORD_SECTION") == 0)
                break;
        }

        while (file >> nodeId >> x >> y)
        {
            // Construct objects in correct memory address, no copy
            vec.emplace_back(static_cast<int>(x), static_cast<int>(y));
            if (line.compare("EOF") == 0)
                break;
        }

        file.close();
    }

    return vec;
}

struct cost_matrix
{
    std::vector<std::vector<uint32_t>> mat;
    size_t problem_size;

    cost_matrix(const std::vector<coords> &v)
    {
        mat = std::vector<std::vector<uint32_t>>(v.size());
        problem_size = v.size();

        for (auto &row : mat)
            row.resize(v.size());

        for (size_t i = 0; i < v.size(); ++i)
        {
            for (size_t j = i + 1; j < v.size(); ++j)
            {
                uint32_t distance = euclidean_distance(v[i], v[j]);
                mat[i][j] = distance;
                mat[j][i] = distance;
            }
        }
    }

    uint32_t &operator[](std::pair<size_t, size_t> &&pair) { return mat[pair.first][pair.second]; }
    const uint32_t &operator[](std::pair<size_t, size_t> &&pair) const { return mat[pair.first][pair.second]; }

    uint32_t compute_cost(std::vector<int> &v) const
    {
        uint32_t cost = mat[v[0]][v[v.size() - 1]];
        for (int i = 1; i < v.size(); ++i)
        {
            cost += mat[v[i - 1]][v[i]];
        }
        return cost;
    }

    uint32_t evaluate_possible_cost(std::vector<int> &v, uint32_t cost_before, size_t x, size_t y) const
    {
        size_t x_prev = x - 1;
        if (x == 0) // Fallback to the vector end
            x_prev = v.size() - 1;

        size_t y_succ = y + 1;
        if (y_succ >= v.size())
            y_succ = 0;

        uint32_t removed_cost = mat[v[x_prev]][v[x]] + mat[v[y]][v[y_succ]];
        uint32_t new_cost = mat[v[x_prev]][v[y]] + mat[v[x]][v[y_succ]];

        if (x_prev == y && x == y_succ)
            return cost_before;
        else
            return cost_before - removed_cost + new_cost;
    }
};

/** Shuffle vector with uniform distribution */
template <typename T>
void shuffle(std::vector<T> &v)
{
    for (size_t i = v.size() - 1; i > 0; --i)
    {
        std::swap(v[random_index(0, i)], v[i]);
    }
}

std::vector<int> random_vector(size_t size)
{
    std::vector<int> v(size);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v);
    return v;
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
optimization_step steepest_optimizer_step(const cost_matrix &mat, std::vector<int> &v, const uint32_t cost)
{
    bool found = false;
    size_t best_from, best_to;

    auto result = optimization_step(cost);
    uint32_t best_cost = cost;

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            uint32_t next_cost = mat.evaluate_possible_cost(v, cost, from, to);
            ++result.seen_solutions;

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

    return result.update_score(best_cost);
}

/** Returns the first best cost and permutated vector v as referrence (if better solution was found)*/
optimization_step greedy_optimizer_step(const cost_matrix &mat, std::vector<int> &v, const uint32_t prev_cost)
{
    auto result = optimization_step(prev_cost);

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            uint32_t next_cost = mat.evaluate_possible_cost(v, prev_cost, from, to);
            ++result.seen_solutions;
            if (next_cost < prev_cost)
            {
                swap_with_rotation(v, from, to);
                return result.update_score(next_cost);
            }
        }
    }

    return result.update_score(prev_cost);
}

auto local_search_optimizer(const std::function<optimization_step(const cost_matrix &mat, std::vector<int> &v, const uint32_t cost)> step)
{
    return [step](const cost_matrix &mat, std::vector<int> &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost, v);

        while (true)
        {
            auto step_result = step(mat, v, result.final_score);
            uint32_t new_cost = step_result.score;
            result.seen_solutions += step_result.seen_solutions;
            ++result.steps;
            if (new_cost < result.final_score)
            {
                result.final_score = new_cost;
            }
            else
            {
                result.final_path = std::vector<int>(v);
                return result;
            }
        }
    };
}

auto time_constrained_optimizer(long limit_ms, const std::function<optimization_step(const cost_matrix &mat, std::vector<int> &v, const uint32_t cost)> step)
{
    return [limit_ms, step](const cost_matrix &mat, std::vector<int> &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost, v);
        result.final_path = std::vector<int>(v);

        long elapsed = 0;
        uint32_t iteration = 0;

        auto start_time = now();
        do
        {
            auto step_result = step(mat, v, cost);
            ++result.seen_solutions;
            ++result.steps;
            if (step_result.score < result.final_score)
            {
                result.final_score = step_result.score;
                result.final_path = std::vector<int>(v);
            }
            elapsed = as_milliseconds(now() - start_time);
        } while (elapsed < limit_ms);

        return result;
    };
}

optimization_step random_step(const cost_matrix &mat, std::vector<int> &v, const uint32_t prev_cost)
{
    shuffle(v);
    return optimization_step(mat.compute_cost(v), 1);
}

optimization_step random_walk_step(const cost_matrix &mat, std::vector<int> &v, const uint32_t prev_cost)
{
    size_t from = random_index(0, v.size() - 1);
    size_t to = from;

    do
    {
        to = random_index(0, v.size() - 1);
    } while (from == to);

    std::swap(v[from], v[to]);

    return optimization_step(mat.compute_cost(v), 1);
}

optimization_result heuristic_optimizer(const cost_matrix &mat, std::vector<int> &v, const uint32_t cost)
{
    optimization_result result(cost, v);

    for (size_t i = 1; i < v.size(); ++i)
    {
        size_t prev_node = i - 1;
        size_t idx_to_swap = i;
        uint32_t prev_cost = mat[{v[i], v[prev_node]}];
        ++result.steps;

        for (size_t j = i + 1; j < v.size(); ++j)
        {
            ++result.seen_solutions;
            uint32_t new_cost = mat[{v[j], v[prev_node]}];
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

    result.final_path = std::vector<int>(v);
    result.final_score = mat.compute_cost(v);
    return result;
}

struct tsp_optimizer
{
    const std::string name;
    const bool shuffle;
    const std::function<optimization_result(const cost_matrix &, std::vector<int> &, const uint32_t)> optimizer;

    tsp_optimizer(std::string name, bool shuffle, std::function<optimization_result(const cost_matrix &, std::vector<int> &, const uint32_t)> optimizer) : name(name), shuffle(shuffle), optimizer(optimizer) {}

    const optimization_result operator()(const cost_matrix &mat, std::vector<int> &v, const uint32_t prev_cost) const
    {
        return optimizer(mat, v, prev_cost);
    }
};

class tsp
{
    std::list<optimization_raport> results;

    const size_t max_iterations;
    const long limit_ms;

    long run_experiment(const cost_matrix &mat, std::string problem, const tsp_optimizer &optimizer)
    {
        auto vector = random_vector(mat.problem_size);
        auto header = measure_time(mat, optimizer, vector);

        size_t iterations = max_iterations;
        if (mat.problem_size > 400)
            iterations = 10;

        for (size_t i = 0; i < iterations; ++i)
        {
            optimization_raport raport(header.with_names(problem, optimizer.name));
            if (optimizer.shuffle)
                shuffle(vector);

            uint32_t cost = mat.compute_cost(vector);
            raport.result = optimizer(mat, vector, cost);

            results.emplace_back(raport);
        }

        return header.time_ms();
    }

    optimization_header measure_time(const cost_matrix &mat, const tsp_optimizer &optimizer, std::vector<int> &v)
    {
        long elapsed = 0;
        uint32_t iteration = 0;

        auto start_time = now();
        do
        {
            if (optimizer.shuffle)
                shuffle(v);

            uint32_t local_cost = mat.compute_cost(v);
            optimizer(mat, v, local_cost);
            ++iteration;
            elapsed = as_milliseconds(now() - start_time);

        } while (elapsed < limit_ms);

        return optimization_header(iteration, elapsed);
    }

public:
    tsp(size_t iterations, long limit_ms) : max_iterations(iterations), limit_ms(limit_ms) {}

    void run_experiments(const cost_matrix &mat, std::string problem)
    {
        // tsp_optimizer("random", false, random_optimizer)};
        long random_limit_ms = run_experiment(mat, problem, tsp_optimizer("steepest", true, local_search_optimizer(steepest_optimizer_step)));
        run_experiment(mat, problem, tsp_optimizer("greedy", true, local_search_optimizer(greedy_optimizer_step)));
        run_experiment(mat, problem, tsp_optimizer("heuristic", true, heuristic_optimizer));
        run_experiment(mat, problem, tsp_optimizer("random", false, time_constrained_optimizer(random_limit_ms, random_step)));
        run_experiment(mat, problem, tsp_optimizer("random_walk", true, time_constrained_optimizer(random_limit_ms, random_walk_step)));
    }

    void add_optimal_tour(const std::string &problem, const std::pair<uint32_t, std::vector<int>> &optimal)
    {
        results.emplace_back(optimization_header(problem, "optimal"), optimization_result(optimal));
    }

    friend std::ostream &operator<<(std::ostream &ostream, const tsp &t)
    {
        ostream << optimization_raport::csv_header << '\n';

        for (auto &raport : t.results)
        {
            ostream << raport;
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

const std::pair<uint32_t, std::vector<int>> optimal_tour(const cost_matrix &mat, std::string path)
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

        uint32_t cost = mat.compute_cost(vec);
        return std::make_pair(cost, vec);
    }
    else
    {
        return std::make_pair(-1, vec);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cout << "Usage: " << argv[0] << "n ms results.csv <file1>.tsp [<file2>.tsp ...]" << '\n';
        return 0;
    }

    size_t n = std::stoul(argv[1]);
    long limit_ms = std::stol(argv[2]);
    std::string output_path(argv[3]);

    auto problems = tsp(n, limit_ms);

    for (int i = 4; i < argc; i++)
    {
        std::string path(argv[i]);
        std::cout << "Processing: " << path << '\n';
        std::string problem = file_name(path);

        auto coords = parse_file(path);
        const auto mat = cost_matrix(coords);
        auto v = random_vector(mat.problem_size);

        auto optimal = optimal_tour(mat, path);
        if (optimal.first < 0)
        {
            std::cout << "Cannot find optimal file for TSP " << path << '\n';
            return -1;
        }

        problems.add_optimal_tour(problem, optimal);
        problems.run_experiments(mat, problem);
    }

    std::ofstream file;
    file.open(output_path);
    file << problems;
    file.close();

    return 0;
}