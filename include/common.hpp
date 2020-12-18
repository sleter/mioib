#ifndef COMMON_HPP
#define COMMON_HPP

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


using path_t = std::vector<int>;

std::string as_string(const path_t &vec);

std::chrono::high_resolution_clock::time_point now();
int64_t as_milliseconds(std::chrono::nanoseconds time);


#endif