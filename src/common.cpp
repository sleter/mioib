#include "common.hpp"

std::string as_string(const path_t &vec)
{
    std::ostringstream os;
    for (auto &i : vec)
        os << i << " ";

    std::string result = os.str();

    if (result.size() > 0)
        result.pop_back();

    return result;
}

std::chrono::high_resolution_clock::time_point now()
{
    return std::chrono::high_resolution_clock::now();
}

int64_t as_milliseconds(std::chrono::nanoseconds time)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
}

void swap_with_rotation(path_t &v, size_t from, size_t to)
{
    if (from > to)
        std::swap(to, from);

    while (to > from)
    {
        std::swap(v[from], v[to]);
        ++from;
        --to;
    }
}

