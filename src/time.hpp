#pragma once

#include<chrono>

std::chrono::high_resolution_clock::time_point now();

int64_t as_milliseconds(std::chrono::nanoseconds time);