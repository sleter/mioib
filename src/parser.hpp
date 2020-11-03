#pragma once

#include<iostream>
#include<fstream>
#include<regex>
#include<math.h>

/** 2D coordinates */
struct coords
{
    int x, y;
    coords(int x, int y);

    friend std::ostream &operator<<(std::ostream &ostream, const coords &c);
};

std::ostream &operator<<(std::ostream &ostream, const coords &c);

long euclidean_distance(const coords &from, const coords &to);

/** 
 * Parse a file to vector of coords. 
 * Uses return value optimization (RVO), no copy 
 * */
std::vector<coords> parse_file(const std::string &path);