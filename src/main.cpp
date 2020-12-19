#include "common.hpp"
#include "results.hpp"
#include "optimizer.hpp"
#include "costs.hpp"
#include "randomutil.hpp"
#include "optimizer_local.hpp"
#include "optimizer_random.hpp"
#include "optimizer_heuristic.hpp"
#include "optimizer_sa.hpp"
#include "optimizer_tabu.hpp"

struct tsp_optimizer
{
    const std::string name;
    const bool shuffle;
    const std::function<optimization_result(const cost_matrix &, path_t &, const uint32_t)> optimizer;

    tsp_optimizer(std::string name, bool shuffle, optimizer_f optimizer) : name(name), shuffle(shuffle), optimizer(optimizer) {}

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
        size_t no_max_change = std::max((size_t)10, mat.problem_size / 20);
        size_t tabu_cadence = std::max((size_t)5, mat.problem_size / 4);
        size_t tabu_elitar_size = std::max((size_t)5, mat.problem_size / 10);

        long steepest_ms = run_experiment(mat, problem, tsp_optimizer("steepest", true, local_search_optimizer(steepest_optimizer_step)));
        long greedy_ms = run_experiment(mat, problem, tsp_optimizer("greedy", true, local_search_optimizer(greedy_optimizer_step)));
        long random_ms = std::max(steepest_ms, greedy_ms);
        
        run_experiment(mat, problem, tsp_optimizer("heuristic", true, heuristic_optimizer));
        run_experiment(mat, problem, tsp_optimizer("random", false, time_constrained_optimizer(random_ms, random_step)));
        run_experiment(mat, problem, tsp_optimizer("random_walk", true, time_constrained_optimizer(random_ms, random_walk_step)));

        run_experiment(mat, problem, tsp_optimizer("sa", true, simulated_anneling_optimizer(0.95, 0.75, no_max_change, 0.90)));
        run_experiment(mat, problem, tsp_optimizer("tabu", true, tabu_optimizer(tabu_cadence, tabu_elitar_size, no_max_change)));
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