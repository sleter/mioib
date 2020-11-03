#pragma once

#include <vector>
#include <string>
#include <list>
#include <sstream>
#include <ostream>

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

template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vector)
{
    os << '[';
    if (vector.size() > 0)
    {
        auto it = vector.begin();
        os << *it;
        ++it;
        for (; it != vector.end(); ++it)
        {
            os << ',' << *it;
        }
    }
    os << ']';
    return os;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const std::list<T> &list)
{
    os << '[';
    if (list.size() > 0)
    {
        auto it = list.begin();
        os << *it;
        ++it;
        for (; it != list.end(); ++it)
            os << ',' << *it;
    }
    os << ']';
    return os;
}