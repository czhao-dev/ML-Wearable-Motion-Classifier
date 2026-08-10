#include "sdl/version.hpp"

#include <benchmark/benchmark.h>

static void BM_VersionString(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(sdl::version_string());
    }
}
BENCHMARK(BM_VersionString);
