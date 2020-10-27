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

std::random_device rd;
std::mt19937 gen(rd());

size_t random_index(size_t from, size_t to)
{
    std::uniform_int_distribution<size_t> distribution(from, to);
    return distribution(gen);
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

std::vector<std::vector<float>> compute_distance_matrix(const std::vector<coords> &v)
{
    auto matrix = std::vector<std::vector<float>>(v.size());
    for (auto &row : matrix)
        row.resize(v.size());

    for (size_t i = 0; i < v.size(); ++i)
    {
        for (size_t j = i + 1; j < v.size(); ++j)
        {
            float distance = euclidean_distance(v[i], v[j]);
            matrix[i][j] = distance;
            matrix[j][i] = distance;
        }
    }

    return matrix;
}

template <typename T>
void print(const std::vector<T> &vec)
{
    for (auto &i : vec)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void print_matrix(const std::vector<std::vector<T>> &mat)
{
    for (auto &row : mat)
    {
        print(row);
    }
    std::cout << std::endl;
}

// Shuffle vector with uniform distribution
template <typename T>
void shuffle(std::vector<T> &v)
{
    for (size_t i = v.size() - 1; i > 0; --i)
    {
        std::swap(v[random_index(0, i)], v[i]);
    }
}

std::vector<int> random_vector(size_t size){
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

float compute_cost(const std::vector<std::vector<float>> &cost_matrix, std::vector<int> &v)
{
    float cost = cost_matrix[v[0]][v[v.size() - 1]];
    // std::cout << "[" << v[v.size() - 1] << "," << v[0] << "]=" << cost_matrix[v[0]][v[v.size() - 1]] << " ";

    for (int i = 1; i < v.size(); ++i)
    {
        // std::cout << "[" << v[i - 1] << "," << v[i] << "]=" << cost_matrix[v[i - 1]][v[i]] << " ";
        cost += cost_matrix[v[i - 1]][v[i]];
    }

    // std::cout << '\n';
    return cost;
}

float evaluate_possible_cost(const std::vector<std::vector<float>> &cost_matrix, std::vector<int> &v, float cost_before, size_t x, size_t y)
{
    size_t x_prev = x - 1;
    if (x == 0) // Fallback to the vector end
        x_prev = v.size() - 1;

    size_t y_succ = y + 1;
    if (y_succ >= v.size())
        y_succ = 0;

    // std::cout << "xp=" << x_prev << ", ys=" << y_succ << '\n';
    // std::cout << "v[x_prev]=" << v[x_prev] << ", v[x]=" << v[x] << ", v[y]=" << v[y] << ", v[y_succ]=" << v[y_succ] << '\n';

    float removed_cost = cost_matrix[v[x_prev]][v[x]] + cost_matrix[v[y]][v[y_succ]];
    float new_cost = cost_matrix[v[x_prev]][v[y]] + cost_matrix[v[x]][v[y_succ]];

    // std::cout << "Possible cost (" << x << "," << y << ")" << '\n'
    //           << "Remove = " << removed_cost << " | [" << v[x_prev] << "," << v[x] << "]=" << cost_matrix[v[x_prev]][v[x]]
    //           << "; [" << v[y] << "," << v[y_succ] << "]=" << cost_matrix[v[y]][v[y_succ]]
    //           << '\n'

    //           << "New = " << new_cost << " | [" << v[x_prev] << "," << v[y] << "]=" << cost_matrix[v[x_prev]][v[y]]
    //           << "; [" << v[x] << "," << v[y_succ] << "]=" << cost_matrix[v[x]][v[y_succ]]
    //           << '\n';

    if (x_prev == y && x == y_succ)
        return cost_before;
    else
        return cost_before - removed_cost + new_cost;
}

/** Returns the best cost and permutated vector v as referrence (if better solution was found)*/
std::pair<float, bool> steepest_optimizer(const std::vector<std::vector<float>> &cost_matrix, std::vector<int> &v, const float cost)
{
    bool found = false;
    size_t best_from, best_to;
    float best_cost = cost;

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            float next_cost = evaluate_possible_cost(cost_matrix, v, cost, from, to);
            if(next_cost < best_cost){
                found = true;
                best_cost = next_cost;
                best_from = from;
                best_to = to;
            }
        }
    }

    if(found){
        swap_with_rotation(v, best_from, best_to);
    }

    return std::make_pair(best_cost, found);
}

void optimize(std::string path){
    auto coords = parse_file(path);
    auto cost_matrix = compute_distance_matrix(coords);
    auto v = random_vector(cost_matrix.size());

    float cost = compute_cost(cost_matrix, v);
    bool improves = true;
    while(improves){
        auto result = steepest_optimizer(cost_matrix, v, cost);
        float new_cost = result.first;

        std::cout << "New cost: " << new_cost <<'\n';
        improves = result.second;
    };

    std::cout << "Best path with distance " << compute_cost(cost_matrix, v) << '\n';
    print(v);
}

std::chrono::high_resolution_clock::time_point now()
{
    return std::chrono::high_resolution_clock::now();
}

int64_t as_milliseconds(std::chrono::nanoseconds time)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
}

int64_t measure_time(std::function<void()> f, long limit_ms)
{
    auto start_time = now();
    uint32_t iteration = 0;
    do
    {
        f();
        ++iteration;
    } while (as_milliseconds(now() - start_time) < limit_ms);
    return as_milliseconds(now() - start_time) / (int64_t)iteration;
}

void test_function()
{
    std::vector<int> v(1000000);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <file>.tsp" << '\n';
        return 0;
    }

    std::string path(argv[1]);
    optimize(path);

    // std::vector<std::vector<float>> mat = {
    //     {0, 1, 2, 3, 4, 5, 6},
    //     {1, 0, 1, 2, 3, 4, 5},
    //     {2, 1, 0, 1, 2, 3, 4},
    //     {3, 2, 1, 0, 1, 2, 3},
    //     {4, 3, 2, 1, 0, 1, 2},
    //     {5, 4, 3, 2, 1, 0, 1},
    //     {6, 5, 4, 3, 2, 1, 0},
    // };

    // std::vector<int> v(7);



    // auto coords = parse_file("tsplib/EUC_2D/a280.tsp");
    // auto mat = compute_distance_matrix(coords);
    // auto v = std::vector<int>(coords.size());

    // std::iota(v.begin(), v.end(), 0);

    // for (int i = 0; i < v.size() - 1; ++i)
    // {
    //     for (int j = i + 1; j < v.size(); ++j)
    //     {
    //         int from = i;
    //         int to = j;

    //         float prev_cost = compute_cost(mat, v);
    //         float possible = evaluate_possible_cost(mat, v, prev_cost, from, to);

    //         print(v);
    //         swap_with_rotation(v, from, to);
    //         print(v);

    //         float new_cost = compute_cost(mat, v);

    //         std::cout << "Prev = " << prev_cost << '\n';
    //         std::cout << possible << " == " << new_cost << '\n';

    //         if (static_cast<int>(abs(possible - new_cost)) > 0)
    //         {
    //             std::cout << "Error!" << '\n';
    //             return 0;
    //         }

    //         std::cout << '\n';
    //     }
    // }

    // if (argc != 2)
    // {
    //     std::cout << "Usage: " << argv[0] << " <file>.tsp" << '\n';
    //     return 0;
    // }

    // std::string path(argv[1]);
    // auto v = parse_file(path);
    // auto m = compute_distance_matrix(v);

    // print_matrix(m);

    // auto v = parse_file("/home/bartek/Repos/mioib/tsplib/EUC_2D/pr2392.tsp");
    // print(v);

    // long total_time = 0;
    // int iters = 10;
    // for(int i=0; i<iters; i++){
    //     total_time += measure_time(test_function, 100);
    // }
    // // 5.048 ms for vector of 5000 and 500 iters
    // std::cout << (double)total_time/iters << " ms" << std::endl;
    return 0;
}