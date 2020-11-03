#include "parser.hpp"

coords::coords(int x, int y) : x(x), y(y) {}

std::ostream &operator<<(std::ostream &ostream, const coords &c)
{
    ostream << "<" << c.x << "," << c.y << ">";
    return ostream;
}

long euclidean_distance(const coords &from, const coords &to)
{
    int dx = from.x - to.x;
    int dy = from.y - to.y;

    float distance = float(dx * dx) + float(dy * dy);
    return static_cast<long>(round(sqrt(distance)));
}

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