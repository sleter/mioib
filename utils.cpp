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

std::default_random_engine rd {static_cast<long unsigned int>(time(0))};
std::mt19937 gen(rd());

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
    if(result.size() > 0) {
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
    return sqrt(distance);
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
        // std::cout << "[" << v[v.size() - 1] << "," << v[0] << "]=" << mat[v[0]][v[v.size() - 1]] << " ";

        for (int i = 1; i < v.size(); ++i)
        {
            // std::cout << "[" << v[i - 1] << "," << v[i] << "]=" << mat[v[i - 1]][v[i]] << " ";
            cost += mat[v[i - 1]][v[i]];
        }

        // std::cout << '\n';
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

        // std::cout << "xp=" << x_prev << ", ys=" << y_succ << '\n';
        // std::cout << "v[x_prev]=" << v[x_prev] << ", v[x]=" << v[x] << ", v[y]=" << v[y] << ", v[y_succ]=" << v[y_succ] << '\n';

        float removed_cost = mat[v[x_prev]][v[x]] + mat[v[y]][v[y_succ]];
        float new_cost = mat[v[x_prev]][v[y]] + mat[v[x]][v[y_succ]];

        // std::cout << "Possible cost (" << x << "," << y << ")" << '\n'
        //           << "Remove = " << removed_cost << " | [" << v[x_prev] << "," << v[x] << "]=" << mat[v[x_prev]][v[x]]
        //           << "; [" << v[y] << "," << v[y_succ] << "]=" << mat[v[y]][v[y_succ]]
        //           << '\n'

        //           << "New = " << new_cost << " | [" << v[x_prev] << "," << v[y] << "]=" << mat[v[x_prev]][v[y]]
        //           << "; [" << v[x] << "," << v[y_succ] << "]=" << mat[v[x]][v[y_succ]]
        //           << '\n';

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
float steepest_optimizer_step(const cost_matrix &mat, std::vector<int> &v, const float cost)
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
float greedy_optimizer_step(const cost_matrix &mat, std::vector<int> &v, const float prev_cost)
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

float base_optimizer(const cost_matrix &mat, std::vector<int> &v, const float cost, std::function<float(const cost_matrix &mat, std::vector<int> &v, const float cost)> step)
{
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
            break;
        }
    };

    return prev_cost;
}

float steepest_optimizer(const cost_matrix &mat, std::vector<int> &v, const float cost)
{
    return base_optimizer(mat, v, cost, steepest_optimizer_step);
}

float greedy_optimizer(const cost_matrix &mat, std::vector<int> &v, const float cost)
{
    return base_optimizer(mat, v, cost, greedy_optimizer_step);
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
    const std::function<float(const cost_matrix &, std::vector<int> &, const float)> optimizer;

    tsp_optimizer(std::string name, std::function<float(const cost_matrix &, std::vector<int> &, const float)> optimizer) : name(name), optimizer(optimizer) {}

    const float operator()(const cost_matrix &mat, std::vector<int> &v, const float prev_cost) const
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
    std::list<std::vector<int>> best_paths;

    const cost_matrix mat;
    const size_t iterations;
    const long limit_ms;

    void measure_time(const tsp_optimizer &optimizer, std::vector<int>& v)
    {
        auto start_time = now();
        long elapsed = 0;
        uint32_t iteration = 0;
        float best_cost = MAXFLOAT;
        
        do
        {
            shuffle(v);
            float local_cost = mat.compute_cost(v);
            float next_cost = optimizer(mat, v, local_cost);

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
        optimizer_names.emplace_back(optimizer.name);
        best_paths.push_back(v);
    }

public:
    tsp(const cost_matrix &mat, size_t iterations, long limit_ms) : mat(mat), iterations(iterations), limit_ms(limit_ms) {}

    void run(const tsp_optimizer &optimizer)
    {
        for (size_t idx = 0; idx < iterations; ++idx)
        {
            std::vector<int> v(mat.mat.size());
            std::iota(v.begin(), v.end(), 0);

            std::cout << "Run " << idx + 1 << '/' << iterations << '\n';
            measure_time(optimizer, v);
        }
    }

    void print_all()
    {
        auto name_it = optimizer_names.begin();
        auto cost_it = best_costs.begin();
        auto time_it = elapsed_time_ms.begin();
        auto loop_it = loops.begin();
        auto path_it = best_paths.begin();

        std::cout << "optimizer_name,best_cost,elapsed_time_ms,loop_count\n";
        while (name_it != optimizer_names.end() && cost_it != best_costs.end() && time_it != elapsed_time_ms.end() && loop_it != loops.end() && path_it != best_paths.end())
        {
            std::cout << *name_it << ',' << *cost_it << ',' << *time_it << ',' << *loop_it << ',' << as_string(*path_it) << '\n';

            ++name_it;
            ++cost_it;
            ++time_it;
            ++loop_it;
            ++path_it;
        }
    }
};

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <file>.tsp <n>" << '\n';
        return 0;
    }

    std::string path(argv[1]);
    size_t n = std::stoul(argv[2]);

    std::list<tsp_optimizer> optimizers = {
        tsp_optimizer("steepest", steepest_optimizer),
        tsp_optimizer("greedy", greedy_optimizer)
        };

    auto coords = parse_file(path);
    const auto mat = cost_matrix(coords);
    auto v = random_vector(mat.mat.size());

    auto problem = tsp(mat, n, 100);

    for (auto &optimizer : optimizers)
    {
        problem.run(optimizer);
    }

    problem.print_all();

    // std::vector<int> v (10);

    // std::iota(v.begin(), v.end(), 0);
    // print(v);
    // shuffle(v);
    // print(v);

    // std::iota(v.begin(), v.end(), 0);
    // print(v);
    // shuffle(v);
    // print(v);
    
    return 0;
}