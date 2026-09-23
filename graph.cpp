#include "graph.h"
#include "const.h"

#include <limits>
#include <queue>
#include <utility>
#include <cmath>

namespace orienteering {

// ============================================================
// Graph 実装
// ============================================================

Graph::Graph(const std::vector<Node>& nodes, const std::vector<Edge>& edges) {
    node_list_ = nodes;
    for (const auto& n : nodes) {
        adj_[n.node_id];  // 空のリストを登録（孤立ノード対策）
    }
    for (const auto& e : edges) {
        adj_[e.from_node].push_back({e.to_node, e.length_m, e.elevation_gain});
    }
}

long long Graph::find_nearest_node(double lat, double lon) const {
    long long best_id = -1;
    double best_dist2 = std::numeric_limits<double>::max();
    for (const auto& n : node_list_) {
        double dlat = n.lat - lat;
        double dlon = n.lon - lon;
        double d2   = dlat * dlat + dlon * dlon;
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_id    = n.node_id;
        }
    }
    return best_id;
}

std::unordered_map<long long, PathInfo> Graph::dijkstra_from(long long src) const {
    // 距離マップ
    std::unordered_map<long long, double>    dist;
    // 親ノード（経路復元用）
    std::unordered_map<long long, long long> parent;

    using State = std::pair<double, long long>;  // (cost, node)
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;

    dist[src]   = 0.0;
    parent[src] = -1;
    pq.push({0.0, src});

    while (!pq.empty()) {
        auto top = pq.top();
        pq.pop();
        double d   = top.first;
        long long u = top.second;
        if (d > dist[u]) continue;

        auto it = adj_.find(u);
        if (it == adj_.end()) continue;

        for (const auto& e : it->second) {
            double nd = d + e.length;
            auto dit  = dist.find(e.to);
            if (dit == dist.end() || nd < dit->second) {
                dist[e.to]   = nd;
                parent[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }

    // 経路復元しながら、各到達ノードまでの累積登りを計算
    std::unordered_map<long long, PathInfo> result;
    for (const auto& kv : dist) {
        long long node    = kv.first;
        double length_sum = kv.second;
        double gain_sum   = 0.0;

        long long cur = node;
        while (parent.at(cur) != -1) {
            long long p = parent.at(cur);
            // p → cur のエッジを見つけて、登りを加算
            for (const auto& e : adj_.at(p)) {
                if (e.to == cur) {
                    gain_sum += e.elevation_gain;
                    break;
                }
            }
            cur = p;
        }
        result[node] = {length_sum, gain_sum, true};
    }
    return result;
}

// ============================================================
// PathCache 実装
// ============================================================

PathCache::PathCache(
    const Graph&                  graph,
    const std::vector<long long>& sources,
    const std::vector<Landmark>&  landmarks)
{
    for (long long src : sources) {
        cache_[src] = graph.dijkstra_from(src);
    }

    euclidean_cache_.assign(
        landmarks.size(), std::vector<double>(landmarks.size(), 0.0));
    for (size_t i = 0; i < landmarks.size(); ++i) {
        for (size_t j = i + 1; j < landmarks.size(); ++j) {
            const double mean_lat = (landmarks[i].lat + landmarks[j].lat) / 2.0;
            const double dlat = (landmarks[i].lat - landmarks[j].lat)
                             * METERS_PER_DEGREE;
            const double dlon = (landmarks[i].lon - landmarks[j].lon)
                             * METERS_PER_DEGREE
                             * std::cos(mean_lat * PI / 180.0);
            const double distance = std::sqrt(dlat * dlat + dlon * dlon);
            euclidean_cache_[i][j] = distance;
            euclidean_cache_[j][i] = distance;
        }
    }
}

PathInfo PathCache::get(long long src, long long dst) const {
    auto sit = cache_.find(src);
    if (sit == cache_.end()) {
        return {PENALTY, PENALTY, false};
    }
    auto dit = sit->second.find(dst);
    if (dit == sit->second.end()) {
        return {PENALTY, PENALTY, false};
    }
    return dit->second;
}

double PathCache::euclidean(int landmark_a, int landmark_b) const {
    if (landmark_a < 0 || landmark_b < 0
        || landmark_a >= static_cast<int>(euclidean_cache_.size())
        || landmark_b >= static_cast<int>(euclidean_cache_.size())) {
        return PENALTY;
    }
    return euclidean_cache_[landmark_a][landmark_b];
}

} // namespace orienteering
