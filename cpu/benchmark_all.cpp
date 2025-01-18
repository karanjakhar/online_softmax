#include <benchmark/benchmark.h>

static void BM_Program1(benchmark::State& state) {
    for (auto _ : state) {
        // Your program1 code here
    }
}
BENCHMARK(BM_Program1);

static void BM_Program2(benchmark::State& state) {
    for (auto _ : state) {
        // Your program2 code here
    }
}
BENCHMARK(BM_Program2);

BENCHMARK_MAIN();
