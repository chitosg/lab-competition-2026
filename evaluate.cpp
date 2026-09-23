#include "evaluate.h"
#include "const.h"

#include <algorithm>
#include <cmath>

namespace orienteering {

// ============================================================
// デコード
// ============================================================
DecodedCourse decode(
    const Chromosome&            chromosome,
    const std::vector<Landmark>& landmarks,
    long long                    gate_node)
{
    DecodedCourse result;
    result.is_valid = false;

    const int N = static_cast<int>(landmarks.size());

    const int n_selected = static_cast<int>(chromosome.size());
    if (n_selected < MIN_CONTROLS || n_selected > MAX_CONTROLS) return result;

    std::vector<char> used(landmarks.size(), 0);
    for (int landmark_idx : chromosome) {
        if (landmark_idx < 0 || landmark_idx >= N || used[landmark_idx]) {
            return result;
        }
        used[landmark_idx] = 1;
    }

    // コース構築
    result.course_nodes.push_back(gate_node);
    for (int landmark_idx : chromosome) {
        result.selected_indices.push_back(landmark_idx);
        result.course_nodes.push_back(landmarks[landmark_idx].nearest_node);
    }
    result.course_nodes.push_back(gate_node);
    result.is_valid = true;
    return result;
}

// ============================================================
// f_map：コントロール数が目標値に近いか
// ============================================================
double f_map(int n_controls) {
    return std::abs(n_controls - Q_TARGET);
}

// ============================================================
// f_dist：コントロール地点が密集していないか
// 緯度経度をメートル換算した簡易ユークリッド距離で評価
// 式(2)どおり、全地点ペアの近接ペナルティを 2/{n(n+1)} で正規化する。
// ============================================================
double f_dist(
    const std::vector<int>&      selected_indices,
    const PathCache&             path_cache)
{
    const int n = static_cast<int>(selected_indices.size());

    if (n < 2) {
        return 0.0;
    }
    double penalty = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double d_ij = path_cache.euclidean(
                selected_indices[i], selected_indices[j]);
            penalty += std::max(0.0, D_MIN - d_ij);
        }
    }

    return 2.0 * penalty / (static_cast<double>(n) * (n + 1));
}

// ============================================================
// f_time：所要時間が 60 分に近いか
// ============================================================
double f_time(double total_distance, double total_gain) {
    double t_walk      = (total_distance / WALK_SPEED) * 60.0;
    double t_climb     = (total_gain     / CLIMB_SPEED) * 60.0;
    double t_estimated = t_walk + t_climb;
    return std::abs(t_estimated - T_TARGET);
}

// ============================================================
// f_route：累積登りが許容値を超えた分だけペナルティ
// ============================================================
double f_route(double total_gain) {
    return std::max(0.0, total_gain - ROUTE_TARGET);
}

// ============================================================
// 単目的スコアへの集約（重み付き和）
// ============================================================
double calc_fitness(const Objectives& obj) {
    return W_MAP   * obj.f_map
         + W_DIST  * obj.f_dist
         + W_TIME  * obj.f_time
         + W_ROUTE * obj.f_route;
}

// ============================================================
// まとめて評価（GA から呼ばれる）
// ============================================================
EvalResult evaluate(
    const Chromosome&            chromosome,
    const std::vector<Landmark>& landmarks,
    const PathCache&             path_cache,
    long long                    gate_node)
{
    EvalResult res;
    res.decoded = decode(chromosome, landmarks, gate_node);

    if (!res.decoded.is_valid) {
        res.is_valid       = false;
        res.fitness        = PENALTY;
        res.objectives     = {PENALTY, PENALTY, PENALTY, PENALTY};
        res.total_distance = PENALTY;
        res.total_gain     = PENALTY;
        return res;
    }

    // 各区間の最短経路をキャッシュから取得し、距離と登りを合計
    double total_distance = 0.0;
    double total_gain     = 0.0;
    bool   has_invalid    = false;

    for (size_t i = 0; i + 1 < res.decoded.course_nodes.size(); ++i) {
        PathInfo p = path_cache.get(
            res.decoded.course_nodes[i],
            res.decoded.course_nodes[i + 1]);
        if (!p.reachable) {
            has_invalid = true;
            break;
        }
        total_distance += p.length;
        total_gain     += p.gain;
    }

    if (has_invalid || total_distance >= PENALTY || total_gain >= PENALTY) {
        res.is_valid       = false;
        res.fitness        = PENALTY;
        res.objectives     = {PENALTY, PENALTY, PENALTY, PENALTY};
        res.total_distance = PENALTY;
        res.total_gain     = PENALTY;
        return res;
    }

    res.total_distance = total_distance;
    res.total_gain     = total_gain;
    res.is_valid       = true;

    res.objectives.f_map   = f_map(static_cast<int>(res.decoded.selected_indices.size()));
    res.objectives.f_dist  = f_dist(res.decoded.selected_indices, path_cache);
    res.objectives.f_time  = f_time(total_distance, total_gain);
    res.objectives.f_route = f_route(total_gain);
    res.fitness            = calc_fitness(res.objectives);
    return res;
}

} // namespace orienteering
