#include "costs.hpp"

uint32_t euclidean_distance(const coords &from, const coords &to)
{
    int64_t dx = from.first - to.first;
    int64_t dy = from.second - to.second;

    int64_t distance = dx * dx + dy * dy;
    return static_cast<uint32_t>(round(sqrt(distance)));
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

cost_matrix::cost_matrix(const std::vector<coords> &v)
{
    mat = std::vector<std::vector<uint32_t>>(v.size());
    problem_size = v.size();

    for (auto &row : mat)
        row.resize(v.size());

    for (size_t i = 0; i < v.size(); ++i)
    {
        for (size_t j = i + 1; j < v.size(); ++j)
        {
            uint32_t distance = euclidean_distance(v[i], v[j]);
            mat[i][j] = distance;
            mat[j][i] = distance;
        }
    }
}

uint32_t &cost_matrix::operator[](std::pair<size_t, size_t> &&pair) { return mat[pair.first][pair.second]; }
const uint32_t &cost_matrix::operator[](std::pair<size_t, size_t> &&pair) const { return mat[pair.first][pair.second]; }

uint32_t cost_matrix::compute_cost(path_t &v) const
{
    uint32_t cost = mat[v[0]][v[v.size() - 1]];
    for (int i = 1; i < v.size(); ++i)
    {
        cost += mat[v[i - 1]][v[i]];
    }
    return cost;
}

uint32_t cost_matrix::evaluate_possible_cost(path_t &v, const uint32_t cost_before, size_t x, size_t y) const
{
    if (x == y)
        return cost_before;
    if (y < x)
        return evaluate_possible_cost(v, cost_before, y, x);

    size_t x_prev = x - 1;
    if (x == 0) // Fallback to the vector end
        x_prev = v.size() - 1;

    size_t y_succ = y + 1;
    if (y_succ >= v.size())
        y_succ = 0;

    uint32_t removed_cost = mat[v[x_prev]][v[x]] + mat[v[y]][v[y_succ]];
    uint32_t new_cost = mat[v[x_prev]][v[y]] + mat[v[x]][v[y_succ]];

    if (x_prev == y && x == y_succ)
        return cost_before;
    else
        return cost_before - removed_cost + new_cost;
}