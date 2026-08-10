add_executable(sdl_tests
    smoke_test.cpp
    generators_test.cpp
    bounded_queue_test.cpp
    p0_pipeline_test.cpp
    index_stream_test.cpp
    transform_test.cpp
    reorder_buffer_test.cpp
    collate_test.cpp
    worker_pool_test.cpp
    mock_batch_consumer_test.cpp
    epoch_shuffle_test.cpp
    sharding_test.cpp
    p2_pipeline_test.cpp
    starvation_test.cpp
)

target_link_libraries(sdl_tests PRIVATE
    sdl
    GTest::gtest_main
    sdl_warnings
    sdl_sanitizers
)

include(GoogleTest)
gtest_discover_tests(sdl_tests)
