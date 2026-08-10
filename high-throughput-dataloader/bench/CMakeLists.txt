add_executable(sdl_bench
    smoke_bench.cpp
    p0_pipeline_bench.cpp
    p1_worker_pool_bench.cpp
    p3_starvation_bench.cpp
    p3_cost_sweep_bench.cpp
)

target_link_libraries(sdl_bench PRIVATE
    sdl
    benchmark::benchmark_main
    sdl_warnings
    sdl_sanitizers
)

# Standalone tool (own main(), not a Google Benchmark file) for isolated
# peak-RSS measurement — see memory_probe.cpp for why it must run as a
# separate process per configuration rather than inside sdl_bench.
add_executable(sdl_memory_probe
    memory_probe.cpp
)

target_link_libraries(sdl_memory_probe PRIVATE
    sdl
    sdl_warnings
    sdl_sanitizers
)
