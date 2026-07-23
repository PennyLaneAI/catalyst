#include <string>

#include <catch2/catch_test_macros.hpp>
#include <infiniband/verbs.h>

#include "Context.hpp"
#include "QpState.hpp"

using namespace rdma::devices::common;

static bool have_rxe()
{
    int n = 0;
    ibv_device **devs = ibv_get_device_list(&n);
    bool found = false;
    for (int i = 0; i < n; ++i)
        if (std::string(ibv_get_device_name(devs[i])) == "rxe0")
            found = true;
    if (devs)
        ibv_free_device_list(devs);
    return found;
}

TEST_CASE("QpState transitions gate the RC bring-up edges", "[common]")
{
    REQUIRE(is_valid_transition(QpState::RESET, QpState::INIT));
    REQUIRE(is_valid_transition(QpState::INIT, QpState::RTR));
    REQUIRE(is_valid_transition(QpState::RTR, QpState::RTS));
    REQUIRE(is_valid_transition(QpState::RTS, QpState::ERROR));
    REQUIRE(is_valid_transition(QpState::RTS, QpState::RESET));
    REQUIRE_FALSE(is_valid_transition(QpState::RESET, QpState::RTR));
    REQUIRE_FALSE(is_valid_transition(QpState::INIT, QpState::RTS));
}

TEST_CASE("Context opens rxe0 with an active port", "[common]")
{
    if (!have_rxe())
        SKIP("no rxe0 RDMA device");
    Context ctx("rxe0");
    REQUIRE(ctx.get() != nullptr);
    ibv_port_attr pa = ctx.port_attr(1);
    REQUIRE(pa.state == IBV_PORT_ACTIVE);
    REQUIRE(pa.active_mtu > 0);
}
