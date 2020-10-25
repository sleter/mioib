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
        for (size_t j = i+1; j < v.size(); ++j)
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

template <typename T>
void swap_with_rotation(std::vector<T> &v, size_t from, size_t to)
{
    while(to > from){
        std::swap(v[from], v[to]);
        ++from;
        --to;
    }
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
    auto v = parse_file(path);
    auto m = compute_distance_matrix(v);

    print_matrix(m);

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