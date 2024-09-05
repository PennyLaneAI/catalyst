#include "gtest/gtest.h"
#include <unistd.h>

namespace {
TEST(Example, example) {
     sleep(10);
     ASSERT_TRUE(true);
}
}
