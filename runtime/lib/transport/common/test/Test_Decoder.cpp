#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "Decoder.hpp"

using namespace rdma::devices::common;

TEST_CASE("EchoDecoder copies min(in_len, out_len) low bytes", "[decoder]")
{
    const std::uint64_t in = 0x0123456789ABCDEFull;
    std::uint64_t out = 0;
    EchoDecoder d;
    d.run(&in, 8, &out, 8);
    REQUIRE(out == in); // full 8-byte echo
    out = 0;
    d.run(&in, 1, &out, 8); // 1-byte syndrome
    REQUIRE(out == 0xEFull);
    out = 0;
    d.run(&in, 8, &out, 2); // 2-byte correction window
    REQUIRE(out == 0xCDEFull);
}
