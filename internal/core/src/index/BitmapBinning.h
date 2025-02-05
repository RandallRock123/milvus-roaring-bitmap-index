#pragma once
#include <vector>
#include <roaring/roaring.hh>
#include "BitmapIndex.h"

namespace milvus {
namespace index {

template<typename T>
class BitmapBinning {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
public:
    static constexpr size_t DEFAULT_BIN_SIZE = 1000;
    
    explicit BitmapBinning(size_t bin_size = DEFAULT_BIN_SIZE) : bin_size_(bin_size) {}
    
    size_t GetBin(size_t permission_id) const {
        return permission_id / bin_size_;
    }
    
    std::pair<size_t, size_t> GetBinRange(size_t bin_id) const {
        size_t start = bin_id * bin_size_;
        size_t end = std::min(start + bin_size_, LARGE_BITMAP_THRESHOLD);
        return {start, end};
    }
    
    roaring::Roaring GetBinMask(size_t bin_id) const {
        roaring::Roaring mask;
        auto [start, end] = GetBinRange(bin_id);
        for (size_t i = start; i < end; ++i) {
            mask.add(i);
        }
        return mask;
    }
    
    std::vector<T> GetBinPermissions(size_t bin_id) const {
        std::vector<T> perms;
        auto [start, end] = GetBinRange(bin_id);
        for (size_t i = start; i < end; ++i) {
            perms.push_back(static_cast<T>(i));
        }
        return perms;
    }
    
    size_t GetNumBins() const {
        return (LARGE_BITMAP_THRESHOLD + bin_size_ - 1) / bin_size_;
    }
    
private:
    size_t bin_size_;
};

} // namespace index
} // namespace milvus
