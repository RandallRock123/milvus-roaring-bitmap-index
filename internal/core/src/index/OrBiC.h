#pragma once
#include <vector>
#include <algorithm>
#include <roaring/roaring.hh>

namespace milvus {
namespace index {

class OrBiC {
public:
    void BuildClusters(const std::vector<roaring::Roaring>& bitmaps) {
        size_t n = bitmaps.size();
        clusters_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            clusters_[i] = std::make_pair(i, bitmaps[i].cardinality());
        }
        std::sort(clusters_.begin(), clusters_.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }
    
    std::vector<size_t> GetOptimizedOrder(const roaring::Roaring& query) {
        std::vector<size_t> order;
        order.reserve(clusters_.size());
        for (const auto& cluster : clusters_) {
            order.push_back(cluster.first);
        }
        return order;
    }
    
private:
    std::vector<std::pair<size_t, size_t>> clusters_;
};

} // namespace index
} // namespace milvus
