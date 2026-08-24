#include <cstddef>

#include "FpgaControllerSession.hpp"
#include "WireProtocol.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace rdma::devices::fpga_verbs;
using namespace catalyst::transport::common;

// Guard the on-wire frame layout the controller's RDMA_WRITE relies on.
TEST_CASE("Wire Payload is the 16 B FPGA frame") {
    STATIC_REQUIRE(sizeof(Payload) == 16);
    STATIC_REQUIRE(offsetof(Payload, value) == 0);
    STATIC_REQUIRE(offsetof(Payload, seq_num) == 8);
    STATIC_REQUIRE(offsetof(Payload, pad) == 12);
}

TEST_CASE("PayloadSlot is a 64 B NIC-aligned ring slot") {
    STATIC_REQUIRE(sizeof(PayloadSlot) == 64);
    STATIC_REQUIRE(alignof(PayloadSlot) == 64);
    STATIC_REQUIRE(K_RING_SLOTS == 256);
    STATIC_REQUIRE(REGION_BYTES == K_RING_SLOTS * sizeof(PayloadSlot));
}

TEST_CASE("FpgaControllerSession is constructible without a device") {
    FpgaControllerSession s("no_such_dev", 3);
    SUCCEED();
}
