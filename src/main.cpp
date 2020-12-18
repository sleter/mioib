#include "common.hpp"
#include "results.hpp"
#include "optimizer.hpp"
#include "costs.hpp"

std::default_random_engine rd{static_cast<long unsigned int>(time(0))};
std::mt19937 gen(rd());

size_t random_index(size_t from, size_t to)
{
    std::uniform_int_distribution<size_t> distribution(from, to);
    return distribution(gen);
}

const float random_float()
{
    std::uniform_real_distribution<float> distribution;
    return distribution(gen);
}

/** Shuffle vector with uniform distribution */
template <typename T>
void shuffle(std::vector<T> &v)
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

template <typename T>
void swap_with_rotation(std::vector<T> &v, size_t from, size_t to)
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

/** Returns the best cost from all possible neighbours and permutated vector v as referrence (if better solution was found)*/
optimization_step steepest_optimizer_step(const cost_matrix &mat, path_t &v, const uint32_t cost)
{
    bool found = false;
    size_t best_from, best_to;

    auto result = optimization_step(cost);
    uint32_t best_cost = cost;

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            uint32_t next_cost = mat.evaluate_possible_cost(v, cost, from, to);
            ++result.seen_solutions;

            if (next_cost < best_cost)
            {
                found = true;
                best_cost = next_cost;
                best_from = from;
                best_to = to;
            }
        }
    }

    if (found)
    {
        swap_with_rotation(v, best_from, best_to);
    }

    return result.update_cost(best_cost);
}

/** Returns the first best cost and permutated vector v as referrence (if better solution was found)*/
optimization_step greedy_optimizer_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost)
{
    auto result = optimization_step(prev_cost);

    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            uint32_t next_cost = mat.evaluate_possible_cost(v, prev_cost, from, to);
            ++result.seen_solutions;
            if (next_cost < prev_cost)
            {
                swap_with_rotation(v, from, to);
                return result.update_cost(next_cost);
            }
        }
    }

    return result.update_cost(prev_cost);
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

struct tabu_elitar_member
{
    size_t from = 0;
    size_t to = 0;
    uint32_t cost = 0;

    tabu_elitar_member() : from(0), to(0), cost(0) {}

    tabu_elitar_member(size_t from, size_t to, uint32_t cost) : from(std::min(from, to)), to(std::max(from, to)), cost(cost) {}

    friend bool operator<(const tabu_elitar_member &left, const tabu_elitar_member &right) { return left.cost < right.cost; }

    friend std::ostream &operator<<(std::ostream &ostream, const tabu_elitar_member &value)
    {
        ostream << "<(" << value.from << "," << value.to << ")," << value.cost << '>';
        return ostream;
    }
};

inline bool operator>(const tabu_elitar_member &left, const tabu_elitar_member &right) { return right < left; }

void push_elitar_member(std::vector<tabu_elitar_member> &elitar_members, const size_t elitar_size, const size_t from, const size_t to, const uint32_t next_cost)
{
    if (elitar_members.size() < elitar_size)
    {
        elitar_members.emplace_back(from, to, next_cost);
        std::push_heap(elitar_members.begin(), elitar_members.end());
    }
    else if (next_cost < elitar_members.front().cost)
    {
        std::pop_heap(elitar_members.begin(), elitar_members.end());
        elitar_members.pop_back();

        elitar_members.emplace_back(from, to, next_cost);
        std::push_heap(elitar_members.begin(), elitar_members.end());
    }
}

void push_tabu_list(std::map<std::pair<size_t, size_t>, size_t> &tabu_list, size_t from, size_t to, size_t initial_cadence)
{
    if (from > to)
        std::swap(from, to);

    auto key = std::make_pair(from, to);
    tabu_list[key] = initial_cadence;
}

void update_tabu_list(std::map<std::pair<size_t, size_t>, size_t> &tabu_list)
{
    for (auto it = tabu_list.begin(); it != tabu_list.end();)
    {
        if (it->second <= 1)
        {
            it = tabu_list.erase(it);
        }
        else
        {
            --it->second;
            ++it;
        }
    }
}

const tabu_elitar_member tabu_find_best(std ::vector<tabu_elitar_member> &elitar_members,
                                        std::map<std::pair<size_t, size_t>, size_t> &tabu_list,
                                        uint32_t best_cost)
{
    auto first_member = elitar_members.front();
    while (elitar_members.size() > 0)
    {
        auto member = elitar_members.front();

        if (member.cost < best_cost)
        {
            return member;
        }
        else if (tabu_list.count({member.from, member.to}) == 0)
        {
            return member;
        }
        else
        {
            std::pop_heap(elitar_members.begin(), elitar_members.end(), std::greater<>{});
            elitar_members.pop_back();
        }
    }
    return first_member;
}

void tabu_generate(const cost_matrix &mat, path_t &v,
                   std::vector<tabu_elitar_member> &elitar_members,
                   optimization_result &result,
                   size_t elitar_size,
                   uint32_t cost)
{
    elitar_members.clear();
    for (size_t from = 0; from < v.size() - 1; ++from)
    {
        for (size_t to = from + 1; to < v.size(); ++to)
        {
            uint32_t next_cost = mat.evaluate_possible_cost(v, cost, from, to);
            push_elitar_member(elitar_members, elitar_size, from, to, next_cost);
            ++result.seen_solutions;
        }
    }
}

auto tabu_optimizer(size_t initial_cadence, size_t elitar_size, size_t max_no_change_iterations)
{
    return [initial_cadence, elitar_size, max_no_change_iterations](const cost_matrix &mat, path_t &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost);
        std::vector<tabu_elitar_member> elitar_members;
        std::map<std::pair<size_t, size_t>, size_t> tabu_list;

        elitar_members.reserve(elitar_size);
        size_t no_improvement_count = 0;
        uint32_t local_best_cost = result.final_cost;

        while (true)
        {
            ++result.steps;
            tabu_generate(mat, v, elitar_members, result, elitar_size, local_best_cost);
            std::make_heap(elitar_members.begin(), elitar_members.end(), std::greater<>{});
            auto local_best = tabu_find_best(elitar_members, tabu_list, result.final_cost);
            local_best_cost = local_best.cost;

            swap_with_rotation(v, local_best.from, local_best.to);
            if (local_best_cost < result.final_cost)
            {
                result.best_solution(local_best_cost, v);
                no_improvement_count = 0;
            }
            else
            {
                ++no_improvement_count;
            }

            update_tabu_list(tabu_list);
            push_tabu_list(tabu_list, local_best.from, local_best.to, initial_cadence);

            if (no_improvement_count >= max_no_change_iterations)
            {
                return result;
            }
        }
    };
}

auto simulated_anneling_optimizer(const float p, const float l_ratio, size_t max_no_change_iterations, float alpha)
{
    return [p, l_ratio, max_no_change_iterations, alpha](const cost_matrix &mat, path_t &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost);

        float mean_next_cost = mean_neighbour_cost(mat, 10000);
        const float c0 = (-(abs(mean_next_cost - cost))) / log(p);

        uint32_t neighbourhood_size = (v.size() * (v.size() - 1)) / 2;
        uint32_t L = static_cast<uint32_t>(round(neighbourhood_size * l_ratio));

        uint32_t current_cost = cost;
        result.final_cost = cost;

        float temperature = c0;
        size_t no_improvement_count = 0;

        while (true)
        {
            ++result.steps;
            uint32_t step_start_cost = result.final_cost;
            for (size_t j = 0; j < L; ++j)
            {
                auto next_solution = random_neighbour(mat, v, current_cost);
                ++result.seen_solutions;
                uint32_t next_cost = std::get<0>(next_solution);
                size_t from = std::get<1>(next_solution);
                size_t to = std::get<2>(next_solution);

                if (next_cost <= current_cost)
                {
                    swap_with_rotation(v, from, to);
                    current_cost = next_cost;
                    if (current_cost < result.final_cost)
                        result.best_solution(current_cost, v);
                }
                else
                {
                    float probability = exp(-((float)next_cost - (float)current_cost) / temperature);
                    if (probability > random_float())
                    {
                        swap_with_rotation(v, from, to);
                        current_cost = next_cost;
                        if (current_cost < result.final_cost)
                            result.best_solution(current_cost, v);
                    }
                }
            }

            temperature *= alpha;
            if (current_cost < step_start_cost)
            {
                no_improvement_count = 0;
            }
            else
            {
                ++no_improvement_count;
            }

            if (no_improvement_count >= max_no_change_iterations && temperature < 0.01)
            {
                break;
            }
        }

        return result;
    };
}

auto local_search_optimizer(const std::function<optimization_step(const cost_matrix &mat, path_t &v, const uint32_t cost)> step)
{
    return [step](const cost_matrix &mat, path_t &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost);

        while (true)
        {
            auto step_result = step(mat, v, result.final_cost);
            uint32_t new_cost = step_result.cost;
            result.seen_solutions += step_result.seen_solutions;
            ++result.steps;
            if (new_cost < result.final_cost)
            {
                result.final_cost = new_cost;
            }
            else
            {
                result.final_path = std::vector<int>(v);
                return result;
            }
        }
    };
}

auto time_constrained_optimizer(long limit_ms, const std::function<optimization_step(const cost_matrix &mat, path_t &v, const uint32_t cost)> step)
{
    return [limit_ms, step](const cost_matrix &mat, path_t &v, const uint32_t cost) -> optimization_result {
        optimization_result result(cost);

        long elapsed = 0;
        uint32_t iteration = 0;

        auto start_time = now();
        do
        {
            auto step_result = step(mat, v, cost);
            ++result.seen_solutions;
            ++result.steps;
            if (step_result.cost < result.final_cost)
            {
                result.final_cost = step_result.cost;
            }
            elapsed = as_milliseconds(now() - start_time);
        } while (elapsed < limit_ms);

        result.final_path = std::vector<int>(v);
        return result;
    };
}

optimization_step random_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost)
{
    shuffle(v);
    return optimization_step(mat.compute_cost(v), 1);
}

optimization_step random_walk_step(const cost_matrix &mat, path_t &v, const uint32_t prev_cost)
{
    auto neighbour = random_neighbour(v);
    return optimization_step(mat.evaluate_possible_cost(v, prev_cost, neighbour.first, neighbour.second), 1);
}

optimization_result heuristic_optimizer(const cost_matrix &mat, path_t &v, const uint32_t cost)
{
    optimization_result result(cost);

    for (size_t i = 1; i < v.size(); ++i)
    {
        size_t prev_node = i - 1;
        size_t idx_to_swap = i;
        uint32_t prev_cost = mat[{v[i], v[prev_node]}];
        ++result.steps;

        for (size_t j = i + 1; j < v.size(); ++j)
        {
            ++result.seen_solutions;
            uint32_t new_cost = mat[{v[j], v[prev_node]}];
            if (new_cost < prev_cost)
            {
                prev_cost = new_cost;
                idx_to_swap = j;
            }
        }
        if (i != idx_to_swap)
        {
            std::swap(v[i], v[idx_to_swap]);
        }
    }

    result.final_path = std::vector<int>(v);
    result.final_cost = mat.compute_cost(v);
    return result;
}

struct tsp_optimizer
{
    const std::string name;
    const bool shuffle;
    const std::function<optimization_result(const cost_matrix &, path_t &, const uint32_t)> optimizer;

    tsp_optimizer(std::string name, bool shuffle, std::function<optimization_result(const cost_matrix &, path_t &, const uint32_t)> optimizer) : name(name), shuffle(shuffle), optimizer(optimizer) {}

    const optimization_result operator()(const cost_matrix &mat, path_t &v, const uint32_t prev_cost) const
    {
        return optimizer(mat, v, prev_cost);
    }
};

class tsp
{
    std::list<optimization_raport> results;

    const size_t max_iterations;
    const long limit_ms;

    long run_experiment(const cost_matrix &mat, std::tuple<std::string, u_int32_t, path_t> &problem, const tsp_optimizer &optimizer)
    {
        std::cout << "\tRunning: " << optimizer.name << " for: " << std::get<0>(problem) <<'\n';
        auto vector = random_vector(mat.problem_size);
        auto header = measure_time(mat, optimizer, vector);

        size_t iterations = max_iterations;
        if (mat.problem_size > 400)
            iterations = std::min(size_t(30), iterations);

        for (size_t i = 0; i < iterations; ++i)
        {
            optimization_raport raport(header.with_names(std::get<0>(problem), vector.size(), optimizer.name, std::get<1>(problem), std::get<2>(problem)));
            if (optimizer.shuffle)
                shuffle(vector);

            uint32_t cost = mat.compute_cost(vector);
            raport.result = optimizer(mat, vector, cost);

            results.emplace_back(raport);
        }

        return header.time_ms();
    }

    optimization_header measure_time(const cost_matrix &mat, const tsp_optimizer &optimizer, path_t &v)
    {
        long elapsed = 0;
        uint32_t iteration = 0;

        auto start_time = now();
        do
        {
            if (optimizer.shuffle)
                shuffle(v);

            uint32_t local_cost = mat.compute_cost(v);
            optimizer(mat, v, local_cost);
            ++iteration;
            elapsed = as_milliseconds(now() - start_time);

        } while (elapsed < limit_ms);

        return optimization_header(iteration, elapsed);
    }

public:
    tsp(size_t iterations, long limit_ms) : max_iterations(iterations), limit_ms(limit_ms) {}

    void add_optimal_tour(const std::string &problem, const uint32_t problem_size, const std::pair<uint32_t, path_t> &optimal)
    {
        results.emplace_back(optimization_header(problem, problem_size, "optimal", optimal.first, optimal.second), optimization_result(optimal.first, optimal.second));
    }

    void run_experiments(const cost_matrix &mat, std::tuple<std::string, u_int32_t, path_t> &&problem)
    {
        long steepest_ms = run_experiment(mat, problem, tsp_optimizer("steepest", true, local_search_optimizer(steepest_optimizer_step)));
        long greedy_ms = run_experiment(mat, problem, tsp_optimizer("greedy", true, local_search_optimizer(greedy_optimizer_step)));
        long random_ms = std::max(steepest_ms, greedy_ms);
        
        run_experiment(mat, problem, tsp_optimizer("heuristic", true, heuristic_optimizer));
        run_experiment(mat, problem, tsp_optimizer("random", false, time_constrained_optimizer(random_ms, random_step)));
        run_experiment(mat, problem, tsp_optimizer("random_walk", true, time_constrained_optimizer(random_ms, random_walk_step)));

        run_experiment(mat, problem, tsp_optimizer("sa", true, simulated_anneling_optimizer(0.95, 0.25, 10, 0.90)));
        // run_experiment(mat, problem, tsp_optimizer("sa_1", true, simulated_anneling_optimizer(0.95, 0.5, 20, 0.90)));
        // run_experiment(mat, problem, tsp_optimizer("sa_2", true, simulated_anneling_optimizer(0.95, 0.75, 10, 0.90)));

        run_experiment(mat, problem, tsp_optimizer("tabu", true, tabu_optimizer(10, 10, 10)));
    }

    friend std::ostream &operator<<(std::ostream &ostream, const tsp &t)
    {
        ostream << optimization_raport::csv_header << '\n';

        for (auto &raport : t.results)
        {
            ostream << raport;
        }
        return ostream;
    }
};

const std::string file_name(const std::string path)
{
    std::regex re("(.*/)?(\\w+)\\.tsp");
    std::smatch matches;
    std::regex_match(path, matches, re);
    return std::string(matches[matches.size() - 1]);
}

const std::pair<uint32_t, path_t> optimal_tour(const cost_matrix &mat, std::string path)
{
    std::string opt_tour = ".opt.tour";
    std::string optimal_solution_path = path;
    optimal_solution_path.replace(optimal_solution_path.find(".tsp"), opt_tour.length(), opt_tour);

    path_t vec;
    std::ifstream file(optimal_solution_path);
    if (!file.good())
        return std::make_pair(0, vec);

    if (file.is_open())
    {
        std::string line;
        int node;

        std::regex re("^(\\w+)\\W*:\\W*(\\w+)$");
        std::smatch matches;

        while (std::getline(file, line))
        {
            // Reserve needed memory
            if (std::regex_match(line, matches, re) && matches[1].compare("DIMENSION") == 0)
                vec.reserve(std::stoi(matches[2]));

            else if (line.compare("TOUR_SECTION") == 0)
                break;
        }

        while (file >> node)
        {
            if (node < 0)
                break;
            vec.emplace_back(node - 1);
        }

        file.close();

        uint32_t cost = mat.compute_cost(vec);
        return std::make_pair(cost, vec);
    }
    else
    {
        return std::make_pair(0, vec);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cout << "Usage: " << argv[0] << " n ms results.csv <file1>.tsp [<file2>.tsp ...]" << '\n';
        return 0;
    }

    size_t n = std::stoul(argv[1]);
    long limit_ms = std::stol(argv[2]);
    std::string output_path(argv[3]);

    auto problems = tsp(n, limit_ms);

    for (int i = 4; i < argc; i++)
    {
        std::string path(argv[i]);
        std::cout << "Processing: " << path << '\n';
        std::string name = file_name(path);

        auto coords = parse_file(path);
        const auto mat = cost_matrix(coords);
        auto v = random_vector(mat.problem_size);

        std::pair<uint32_t, path_t> optimal = optimal_tour(mat, path);
        if (optimal.first == 0)
        {
            std::cout << "Cannot find optimal file for TSP " << path << '\n';
            return -1;
        }

        problems.add_optimal_tour(name, v.size(), optimal);
        problems.run_experiments(mat, {name, optimal.first, optimal.second});
    }

    std::ofstream file;
    file.open(output_path);
    file << problems;
    file.close();

    return 0;
}