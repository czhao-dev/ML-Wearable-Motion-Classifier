add_library(sdl
    version.cpp
    generators/seed_util.cpp
    generators/cost_knob.cpp
    generators/image_generator.cpp
    generators/token_generator.cpp
    generators/tabular_generator.cpp
    pipeline/producer.cpp
    pipeline/mock_consumer.cpp
    pipeline/index_stream.cpp
    pipeline/worker_pool.cpp
    pipeline/mock_batch_consumer.cpp
    pipeline/epoch_shuffle.cpp
    pipeline/sharding.cpp
    transform/transform_pipeline.cpp
    transform/scale_transform.cpp
    collate/collate.cpp
)

target_include_directories(sdl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(sdl PUBLIC Threads::Threads)
target_link_libraries(sdl PRIVATE sdl_warnings sdl_sanitizers)
target_compile_features(sdl PUBLIC cxx_std_20)
