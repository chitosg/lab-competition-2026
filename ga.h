#ifndef GA_H
#define GA_H

#include "csv_loader.h"
#include "graph.h"
#include "evaluate.h"

#include <random>
#include <utility>
#include <vector>

namespace orienteering {

using RNG = std::mt19937;

// ============================================================
// GA 操作（個体生成・選択・交叉・突然変異）
// ============================================================

// ランダムな染色体を生成（初期個体群用）
Chromosome create_random_chromosome(int N, RNG& rng);

// トーナメント選択
const Chromosome& tournament_select(
    const std::vector<Chromosome>& population,
    const std::vector<double>&     fitnesses,
    RNG&                           rng);

// 交叉（共通コントロールを必ず継承する集合交叉）
std::pair<Chromosome, Chromosome> crossover(
    const Chromosome& parent1,
    const Chromosome& parent2,
    int               N,
    RNG&              rng);

// 突然変異（追加・削除・置換）
void mutate(Chromosome& chromosome, int N, RNG& rng);

// ============================================================
// GA メインループ
// ============================================================

struct GAResult {
    Chromosome           best_chromosome;
    EvalResult           best_eval;
    std::vector<double>  best_fitness_history;
};

GAResult run_ga(
    const std::vector<Landmark>& landmarks,
    const PathCache&             path_cache,
    long long                    gate_node,
    RNG&                         rng);

} // namespace orienteering

#endif // GA_H
