#include "randomutil.hpp"

std::default_random_engine rd {static_cast<long unsigned int>(time(0))};
std::mt19937 gen (rd());

size_t random_index(size_t from, size_t to)
{
    std::uniform_int_distribution<size_t> distribution(from, to);
    return distribution(gen);
}

float random_float()
{
    std::uniform_real_distribution<float> distribution;
    return distribution(gen);
}

void shuffle(path_t &v)
{
    for (size_t i = v.size() - 1; i > 0; --i)
    {
        std::swap(v[random_index(0, i)], v[i]);
    }
}

path_t random_vector(size_t size)
{
    path_t v(size);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v);
    return v;
}

std::pair<size_t, size_t> random_neighbour(path_t &v)
{
    size_t from = random_index(0, v.size() - 1);
    size_t to = from;

    do
    {
        to = random_index(0, v.size() - 1);
    } while (from == to);

    if (from > to)
        std::swap(from, to);
    return {from, to};
}

std::tuple<uint32_t, size_t, size_t> random_neighbour(const cost_matrix &mat, path_t &v, const uint32_t cost)
{
    auto neighbour = random_neighbour(v);
    uint32_t next_cost = mat.evaluate_possible_cost(v, cost, neighbour.first, neighbour.second);
    return {next_cost, neighbour.first, neighbour.second};
}

float mean_neighbour_cost(const cost_matrix &mat, size_t samples)
{
    uint64_t diffs = 0;

    for (size_t i = 0; i < samples; ++i)
    {
        auto v = random_vector(mat.problem_size);
        auto cost = mat.compute_cost(v);
        auto neighbour = random_neighbour(v);
        auto next_cost = mat.evaluate_possible_cost(v, cost, neighbour.first, neighbour.second);

        diffs += abs((long)cost - (long)next_cost);
    }

    return diffs / (float)samples;
}