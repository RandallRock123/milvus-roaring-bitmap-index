#include <benchmark/benchmark.h>
#include <random>
#include "index/BitmapIndex.h"

using namespace milvus::index;

#include <benchmark/benchmark.h>
#include <random>
#include "index/BitmapIndex.h"

using namespace milvus::index;

static void BM_BitmapRange(benchmark::State& state) {
    BitmapIndex<int64_t> index;
    const size_t num_rows = LARGE_BITMAP_THRESHOLD;
    std::vector<int64_t> values(num_rows);
    std::iota(values.begin(), values.end(), 0);
    index.Build(values.data(), num_rows);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, num_rows - 1);
    
    for (auto _ : state) {
        state.PauseTiming();
        int64_t target = dist(gen);
        state.ResumeTiming();
        
        auto result = index.Range(target, OpType::GreaterEqual);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_BitmapRange);

static void BM_BitmapRangeInclusive(benchmark::State& state) {
    BitmapIndex<int64_t> index;
    const size_t num_rows = LARGE_BITMAP_THRESHOLD;
    std::vector<int64_t> values(num_rows);
    std::iota(values.begin(), values.end(), 0);
    index.Build(values.data(), num_rows);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, num_rows - 1);
    
    for (auto _ : state) {
        state.PauseTiming();
        int64_t lower = dist(gen);
        int64_t upper = std::min(lower + num_rows/10, num_rows - 1);
        state.ResumeTiming();
        
        auto result = index.Range(lower, true, upper, true);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_BitmapRangeInclusive);

static void BM_BitmapInOperation(benchmark::State& state) {
    BitmapIndex<int64_t> index;
    const size_t num_rows = LARGE_BITMAP_THRESHOLD;
    std::vector<int64_t> values(num_rows);
    std::iota(values.begin(), values.end(), 0);
    index.Build(values.data(), num_rows);
    
    std::vector<int64_t> query_values;
    query_values.reserve(1000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, num_rows - 1);
    
    for (int i = 0; i < 1000; ++i) {
        query_values.push_back(dist(gen));
    }
    
    for (auto _ : state) {
        auto result = index.In(query_values.size(), query_values.data());
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_BitmapInOperation);

BENCHMARK_MAIN();
