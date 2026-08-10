#include "sdl/version.hpp"

#include <gtest/gtest.h>

TEST(Smoke, VersionStringIsWellFormed) {
    EXPECT_EQ(sdl::version_string(), "0.1.0");
}
