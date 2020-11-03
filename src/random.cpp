#include "random.hpp"

random_utils::random_utils()
{
    engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    generator = std::mt19937(engine());
}

const size_t random_utils::index(const size_t from, const size_t to)
{
    std::uniform_int_distribution<size_t> distribution(from, to);
    return distribution(generator);
}

void random_utils::shuffle(std::vector<int> &v)
{
    for (size_t i = v.size() - 1; i > 0; --i)
    {
        std::swap(v[index(0, i)], v[i]);
    }
}

std::vector<int> random_utils::vector(const size_t size)
{
    std::vector<int> v(size);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v);
    return v;
}
