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

struct optimization_header
{
    std::string problem;
    std::string optimizer;
    size_t time_iterations;
    long time_duration_ms;

    long time_ms() const
    {
        if (time_iterations <= 0 || time_iterations <= 0)
            return 1;
        else
            return time_duration_ms / time_iterations;
    }
};

struct optimization_step
{
    size_t seen_solutions;
    float score;
};

struct optimization_result
{
    float start_score;
    float final_score;
    std::vector<int> start_path;
    std::vector<int> final_path;
    size_t steps;
    size_t seen_solutions;
};

struct optimization_raport
{
    optimization_header header;
    optimization_result result;

    optimization_raport next_raport()
    {
        optimization_raport raport;
        raport.header = header;
        return raport;
    }

    std::string csv_header()
    {
        return "problem,optimizer,start_score,final_score,start_path,final_path,time_duration_ms,time_iterations,time_ms,steps,seen_solutions\n";
    }

    friend std::ostream &operator<<(std::ostream &ostream, const optimization_raport &r)
    {
        ostream << r.header.problem << ',' << r.header.optimizer << ',' << r.result.start_score << ','
                << r.result.final_score << ',' << as_string(r.result.start_path) << ',' << as_string(r.result.final_path) << ','
                << r.header.time_duration_ms << ',' << r.header.time_iterations << ','
                << r.header.time_ms() << ',' << r.result.steps << ',' << r.result.seen_solutions << '\n';
        return ostream;
    }
};

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

struct coords
{
    int x, y;

    coords(int x, int y) : x(x), y(y) {}

    friend std::ostream &operator<<(std::ostream &ostream, const coords &c)
    {
        ostream << "<" << c.x << "," << c.y << ">";
        return ostream;
    }
};

float euclidean_distance(const coords &from, const coords &to)
{
    int dx = from.x - to.x;
    int dy = from.y - to.y;

    float distance = float(dx * dx) + float(dy * dy);
    return round(sqrt(distance));
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
        float x, y;

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
    std::vector<std::vector<float>> mat;

    cost_matrix(const std::vector<coords> &v)
    {
        mat = std::vector<std::vector<float>>(v.size());
        for (auto &row : mat)
            row.resize(v.size());

        for (size_t i = 0; i < v.size(); ++i)
        {
            for (size_t j = i + 1; j < v.size(); ++j)
            {
                float distance = euclidean_distance(v[i], v[j]);
                mat[i][j] = distance;
                mat[j][i] = distance;
            }
        }
    }

    std::vector<float> &operator[](size_t idx) { return mat[idx]; }
    const std::vector<float> &operator[](size_t idx) const { return mat[idx]; }

    float compute_cost(std::vector<int> &v) const
    {
        float cost = mat[v[0]][v[v.size() - 1]];
        for (int i = 1; i < v.size(); ++i)
        {
            cost += mat[v[i - 1]][v[i]];
        }
        return cost;
    }

    float evaluate_possible_cost(std::vector<int> &v, float cost_before, size_t x, size_t y) const
    {
        size_t x_prev = x - 1;
        if (x == 0) // Fallback to the vector end
            x_prev = v.size() - 1;

        size_t y_succ = y + 1;
        if (y_succ >= v.size())
            y_succ = 0;

        float removed_cost = mat[v[x_prev]][v[x]] + mat[v[y]][v[y_succ]];
        float new_cost = mat[v[x_prev]][v[y]] + mat[v[x]][v[y_succ]];

        if (x_prev == y && x == y_succ)
            return cost_before;
        else
            return cost_before - removed_cost + new_cost;
    }

    void print_matrix()
    {
        for (auto &row : mat)
        {
            print(row);
        }
        std::cout << std::endl;
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
optimization_step steepest_optimizer_step(const cost_matrix &mat, std::vector<int> &v, const float cost)
{
    bool found = false;
    size_t best_from, best_to;
    optimization_step result;
    result.seen_solutions = 0;
    float best_cost = cost;

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            float next_cost = mat.evaluate_possible_cost(v, cost, from, to);
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

    result.score = best_cost;
    return result;
}

/** Returns the first best cost and permutated vector v as referrence (if better solution was found)*/
optimization_step greedy_optimizer_step(const cost_matrix &mat, std::vector<int> &v, const float prev_cost)
{
    optimization_step result;
    result.seen_solutions = 0;

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            float next_cost = mat.evaluate_possible_cost(v, prev_cost, from, to);
            ++result.seen_solutions;
            if (next_cost < prev_cost)
            {
                swap_with_rotation(v, from, to);
                result.score = next_cost;
                return result;
            }
        }
    }

    result.score = prev_cost;
    return result;
}

auto local_search_optimizer(const std::function<optimization_step(const cost_matrix &mat, std::vector<int> &v, const float cost)> step)
{
    return [step](const cost_matrix &mat, std::vector<int> &v, const float cost) -> optimization_result {
        optimization_result result;
        result.seen_solutions = 0;
        result.start_score = cost;
        result.start_path = std::vector<int>(v);
        float prev_cost = cost;
        size_t steps = 0;

        while (true)
        {
            auto step_result = step(mat, v, prev_cost);
            float new_cost = step_result.score;
            result.seen_solutions += step_result.seen_solutions;
            ++steps;
            if (new_cost < prev_cost)
            {
                prev_cost = new_cost;
            }
            else
            {
                result.final_path = std::vector<int>(v);
                result.final_score = prev_cost;
                result.steps = steps;
                return result;
            }
        }
    };
}

auto time_constrained_optimizer(long limit_ms, const std::function<optimization_step(const cost_matrix &mat, std::vector<int> &v, const float cost)> step)
{
    return [limit_ms, step](const cost_matrix &mat, std::vector<int> &v, const float cost) -> optimization_result {
        optimization_result result;
        result.final_path= std::vector<int>(v);
        result.final_score = cost;
        result.start_path = std::vector<int>(v);
        result.start_score = cost;
        result.steps = 0;
        result.seen_solutions = 0;

        long elapsed = 0;
        uint32_t iteration = 0;

        float best_cost = cost;
        auto start_time = now();
        do
        {
            auto step_result = step(mat, v, cost);
            ++result.seen_solutions;
            ++result.steps;
            if(step_result.score < best_cost)
            {
                best_cost=step_result.score;
                result.final_path = std::vector<int>(v);
                result.final_score = best_cost;
            }
            elapsed = as_milliseconds(now() - start_time);
        } while (elapsed < limit_ms);

        return result;
    };
}

optimization_step random_step(const cost_matrix &mat, std::vector<int> &v, const float prev_cost){
    optimization_step step;
    shuffle(v);

    step.seen_solutions = 1;
    step.score = mat.compute_cost(v);
    return step;
}

optimization_step random_walk_step(const cost_matrix &mat, std::vector<int> &v, const float prev_cost){
    optimization_step step;
    size_t from = random_index(0, v.size()-1);
    size_t to = from;

    do {
        to = random_index(0, v.size()-1);
    } while (from == to);
    
    std::swap(v[from], v[to]);

    step.seen_solutions = 1;
    step.score = mat.compute_cost(v);
    return step;
}

optimization_result heuristic_optimizer(const cost_matrix &mat, std::vector<int> &v, const float cost)
{
    optimization_result result;
    result.start_path = std::vector<int>(v);
    result.start_score = cost;
    result.seen_solutions = 0;
    result.steps = 0;

    for (size_t i = 1; i < v.size(); ++i)
    {
        size_t prev_node = i - 1;
        size_t idx_to_swap = i;
        float prev_cost = mat[v[i]][v[prev_node]];
        ++result.steps;

        for (size_t j = i + 1; j < v.size(); ++j)
        {
            ++result.seen_solutions;
            float new_cost = mat[v[j]][v[prev_node]];
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
    const std::function<optimization_result(const cost_matrix &, std::vector<int> &, const float)> optimizer;

    tsp_optimizer(std::string name, bool shuffle, std::function<optimization_result(const cost_matrix &, std::vector<int> &, const float)> optimizer) : name(name), shuffle(shuffle), optimizer(optimizer) {}

    const optimization_result operator()(const cost_matrix &mat, std::vector<int> &v, const float prev_cost) const
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
        auto vector = random_vector(mat.mat.size());
        auto header = measure_time(mat, optimizer, vector);
        header.optimizer = optimizer.name;
        header.problem = problem;

        optimization_raport raport;
        raport.header = header;

        size_t iterations = max_iterations;
        if(mat.mat.size() > 400) iterations=10;

        for (size_t i = 0; i < iterations; ++i)
        {
            auto local_raport = raport.next_raport();
            if (optimizer.shuffle)
                shuffle(vector);

            float cost = mat.compute_cost(vector);
            local_raport.result = optimizer.optimizer(mat, vector, cost);
            results.emplace_back(local_raport);
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

            float local_cost = mat.compute_cost(v);
            optimizer(mat, v, local_cost);
            ++iteration;
            elapsed = as_milliseconds(now() - start_time);

        } while (elapsed < limit_ms);

        optimization_header header;
        header.time_iterations = iteration;
        header.time_duration_ms = elapsed;
        return header;
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

    void add_optimal_tour(std::string problem, std::pair<float, std::vector<int>> &optimal)
    {
        optimization_raport raport;
        raport.header.optimizer = "optimal";
        raport.header.problem = problem;
        raport.header.time_duration_ms = 0;
        raport.header.time_iterations = 0;
        raport.result.start_path = optimal.second;
        raport.result.final_path = optimal.second;
        raport.result.steps = 0;
        raport.result.start_score = optimal.first;
        raport.result.final_score = optimal.first;
        raport.result.seen_solutions = 0;

        results.push_back(raport);
    }

    friend std::ostream &operator<<(std::ostream &ostream, const tsp &t)
    {
        optimization_raport r;
        ostream << r.csv_header();

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
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << "n ms <file1>.tsp [<file2>.tsp ...]" << '\n';
        return 0;
    }

    size_t n = std::stoul(argv[1]);
    long limit_ms = std::stol(argv[2]);

    auto problems = tsp(n, limit_ms);

    for (int i = 3; i < argc; i++)
    {
        std::string path(argv[i]);
        std::cout << "Processing: " << path << '\n';
        std::string problem = file_name(path);

        auto coords = parse_file(path);
        const auto mat = cost_matrix(coords);
        auto v = random_vector(mat.mat.size());

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
    file.open("results.csv");
    file << problems;
    file.close();

    return 0;
}