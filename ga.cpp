#include "ga.h"
#include "const.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>

namespace orienteering {

Chromosome create_random_chromosome(int N, RNG& rng) {
    std::uniform_int_distribution<int> dist_n(MIN_CONTROLS, MAX_CONTROLS);
    const int n_select = dist_n(rng);
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(n_select);
    return indices;
}

// ============================================================
// トーナメント選択
// ============================================================
const Chromosome& tournament_select(
    const std::vector<Chromosome>& population,
    const std::vector<double>&     fitnesses,
    RNG&                           rng)
{
    std::uniform_int_distribution<int> dist(0, static_cast<int>(population.size()) - 1);
    int    best     = -1;
    double best_fit = std::numeric_limits<double>::max();
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        int idx = dist(rng);
        if (fitnesses[idx] <= best_fit) {
            best_fit = fitnesses[idx];
            best     = idx;
        }
    }
    return population[best];
}

std::pair<Chromosome, Chromosome> crossover(
    const Chromosome& parent1,
    const Chromosome& parent2,
    int               N,
    RNG&              rng)
{
    std::uniform_real_distribution<double> ureal(0.0, 1.0);
    auto make_child = [&](const Chromosome& first,
                          const Chromosome& second) {
        Chromosome child;
        std::vector<char> used(N, 0);
        for (int value : first) {
            const bool common = std::find(second.begin(), second.end(), value)
                             != second.end();
            if (common || ureal(rng) < 0.5) {
                child.push_back(value);
                used[value] = 1;
            }
        }
        for (int value : second) {
            if (!used[value] && ureal(rng) < 0.5) {
                child.push_back(value);
                used[value] = 1;
            }
        }

        std::vector<int> unused;
        for (int i = 0; i < N; ++i) if (!used[i]) unused.push_back(i);
        std::shuffle(unused.begin(), unused.end(), rng);
        while (static_cast<int>(child.size()) < MIN_CONTROLS) {
            child.push_back(unused.back());
            unused.pop_back();
        }
        while (static_cast<int>(child.size()) > MAX_CONTROLS) {
            std::uniform_int_distribution<int> remove_dist(
                0, static_cast<int>(child.size()) - 1);
            child.erase(child.begin() + remove_dist(rng));
        }
        std::shuffle(child.begin(), child.end(), rng);
        return child;
    };

    return {make_child(parent1, parent2), make_child(parent2, parent1)};
}

void mutate(Chromosome& chromosome, int N, RNG& rng) {
    std::uniform_real_distribution<double> ureal(0.0, 1.0);
    if (ureal(rng) >= 0.12) return;

    std::vector<char> selected(N, 0);
    for (int value : chromosome) selected[value] = 1;
    std::vector<int> unused;
    for (int i = 0; i < N; ++i) if (!selected[i]) unused.push_back(i);

    std::uniform_int_distribution<int> operation(0, 2);
    const int action = operation(rng);
    if (action == 0 && chromosome.size() < MAX_CONTROLS && !unused.empty()) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(unused.size()) - 1);
        chromosome.push_back(unused[pick(rng)]);
    } else if (action == 1 && chromosome.size() > MIN_CONTROLS) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(chromosome.size()) - 1);
        chromosome.erase(chromosome.begin() + pick(rng));
    } else if (action == 2 && !chromosome.empty() && !unused.empty()) {
        std::uniform_int_distribution<int> selected_pick(0, static_cast<int>(chromosome.size()) - 1);
        std::uniform_int_distribution<int> unused_pick(0, static_cast<int>(unused.size()) - 1);
        chromosome[selected_pick(rng)] = unused[unused_pick(rng)];
    }
}

namespace {

double route_cost(
    const Chromosome& route,
    const std::vector<Landmark>& landmarks,
    const PathCache& path_cache,
    long long gate_node)
{
    double cost = 0.0;
    long long previous = gate_node;
    for (int index : route) {
        const PathInfo path = path_cache.get(previous, landmarks[index].nearest_node);
        if (!path.reachable) return PENALTY;
        cost += path.length / WALK_SPEED * 60.0
              + path.gain / CLIMB_SPEED * 60.0;
        previous = landmarks[index].nearest_node;
    }
    const PathInfo back = path_cache.get(previous, gate_node);
    if (!back.reachable) return PENALTY;
    return cost + back.length / WALK_SPEED * 60.0
               + back.gain / CLIMB_SPEED * 60.0;
}

void two_opt(
    Chromosome& route,
    const std::vector<Landmark>& landmarks,
    const PathCache& path_cache,
    long long gate_node)
{
    bool improved = true;
    while (improved) {
        improved = false;
        const double current = route_cost(route, landmarks, path_cache, gate_node);
        for (size_t first = 0; first + 1 < route.size() && !improved; ++first) {
            for (size_t last = first + 1; last < route.size(); ++last) {
                std::reverse(route.begin() + first, route.begin() + last + 1);
                const double candidate = route_cost(route, landmarks, path_cache, gate_node);
                if (candidate + 1e-9 < current) {
                    improved = true;
                    break;
                }
                std::reverse(route.begin() + first, route.begin() + last + 1);
            }
        }
    }
}

} // namespace

// ============================================================
// GA メインループ
// ============================================================
GAResult run_ga(
    const std::vector<Landmark>& landmarks,
    const PathCache&             path_cache,
    long long                    gate_node,
    RNG&                         rng)
{
    const int N = static_cast<int>(landmarks.size());

    std::vector<Chromosome> population;
    population.reserve(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i) {
        population.push_back(create_random_chromosome(N, rng));
        two_opt(population.back(), landmarks, path_cache, gate_node);
    }

    // 初期評価
    std::vector<double> fitnesses(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i) {
        fitnesses[i] = evaluate(population[i], landmarks, path_cache, gate_node).fitness;
    }

    std::vector<double> best_history;
    best_history.reserve(N_GEN);

    for (int gen = 1; gen <= N_GEN; ++gen) {
        std::vector<Chromosome> next_pop;
        next_pop.reserve(POP_SIZE);

        std::vector<int> elite_order(POP_SIZE);
        std::iota(elite_order.begin(), elite_order.end(), 0);
        std::partial_sort(elite_order.begin(), elite_order.begin() + 2,
                          elite_order.end(), [&](int a, int b) {
                              return fitnesses[a] < fitnesses[b];
                          });
        next_pop.push_back(population[elite_order[0]]);
        next_pop.push_back(population[elite_order[1]]);

        // 残りは選択・交叉・突然変異で生成
        while (static_cast<int>(next_pop.size()) < POP_SIZE) {
            const Chromosome& p1 = tournament_select(population, fitnesses, rng);
            const Chromosome& p2 = tournament_select(population, fitnesses, rng);
            auto children = crossover(p1, p2, N, rng);
            mutate(children.first,  N, rng);
            mutate(children.second, N, rng);
            two_opt(children.first, landmarks, path_cache, gate_node);
            two_opt(children.second, landmarks, path_cache, gate_node);
            next_pop.push_back(std::move(children.first));
            if (static_cast<int>(next_pop.size()) < POP_SIZE) {
                next_pop.push_back(std::move(children.second));
            }
        }

        population = std::move(next_pop);

        // 新世代を評価
        for (int i = 0; i < POP_SIZE; ++i) {
            fitnesses[i] = evaluate(population[i], landmarks, path_cache, gate_node).fitness;
        }

        double best = *std::min_element(fitnesses.begin(), fitnesses.end());
        best_history.push_back(best);

        std::cout << "  [世代 " << gen << "]  best_fitness = " << best << std::endl;
    }

    int best_idx = static_cast<int>(
        std::min_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin());

    GAResult result;
    result.best_chromosome      = population[best_idx];
    result.best_eval            = evaluate(result.best_chromosome, landmarks, path_cache, gate_node);
    result.best_fitness_history = std::move(best_history);
    return result;
}

} // namespace orienteering
