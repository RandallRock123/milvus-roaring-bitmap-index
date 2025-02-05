#pragma once
#include <vector>
#include <roaring/roaring.hh>

namespace milvus {
namespace index {

class BitmapBinning {
public:
    static constexpr size_t DEFAULT_BIN_SIZE = 1000;
    
    explicit BitmapBinning(size_t bin_size = DEFAULT_BIN_SIZE) : bin_size_(bin_size) {}
    
    size_t GetBin(size_t permission_id) const {
        return permission_id / bin_size_;
    }
    
    std::vector<size_t> GetBinPermissions(size_t bin_id) const {
        std::vector<size_t> permissions;
        size_t start = bin_id * bin_size_;
        size_t end = std::min(start + bin_size_, LARGE_BITMAP_THRESHOLD);
        permissions.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            permissions.push_back(i);
        }
        return permissions;
    }
    
private:
    size_t bin_size_;
};

} // namespace index
} // namespace milvus
