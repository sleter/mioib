#include "optimizer_tabu.hpp"

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

bool should_tabu_regenerate_elitars(const cost_matrix &mat, path_t &v,
                                    std::vector<tabu_elitar_member> &elitar_members,
                                    uint32_t cost)
{
    auto best_member = elitar_members.front();
    auto possible_cost = mat.evaluate_possible_cost(v, cost, best_member.from, best_member.to);
    float size_ratio = elitar_members.size() / (float)(elitar_members.capacity());

    // Regenerate elitar list, if the best elitar is worst then the current solution
    // Or elitar members has only 25% of thier size
    return (possible_cost >= cost || size_ratio < 0.25);
}

void tabu_generate(const cost_matrix &mat, path_t &v,
                   std::vector<tabu_elitar_member> &elitar_members,
                   optimization_result &result,
                   size_t elitar_size,
                   uint32_t cost)
{
    if (elitar_members.empty() || should_tabu_regenerate_elitars(mat, v, elitar_members, cost))
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
}

optimizer_f tabu_optimizer(size_t initial_cadence, size_t elitar_size, size_t max_no_change_iterations)
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