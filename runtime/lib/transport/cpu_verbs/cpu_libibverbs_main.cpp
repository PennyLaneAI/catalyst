#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "CpuControllerSession.hpp"
#include "CpuCoprocessorSession.hpp"
#include "WireProtocol.hpp"

using namespace catalyst::transport;
using namespace rdma::devices::cpu_libibverbs;
using namespace rdma::devices::common; // REGION_BYTES, DEMO_SYNDROME, Payload

int main(int argc, char **argv)
{
    std::string role = "coprocessor", dev = "rxe0", peer = "127.0.0.1";
    int gid = 1;
    std::uint16_t port = 18560;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--role")
            role = v;
        else if (k == "--dev")
            dev = v;
        else if (k == "--gid")
            gid = std::atoi(v.c_str());
        else if (k == "--peer")
            peer = v;
        else if (k == "--port")
            port = static_cast<std::uint16_t>(std::atoi(v.c_str()));
    }
    const bool is_coprocessor = (role == "coprocessor");

    std::unique_ptr<TransportSession> s;
    CpuCoprocessorSession *coproc = nullptr;
    CpuControllerSession *controller = nullptr;
    if (is_coprocessor) {
        auto up = std::make_unique<CpuCoprocessorSession>(dev, gid);
        coproc = up.get();
        s = std::move(up);
    }
    else {
        auto up = std::make_unique<CpuControllerSession>(dev, gid);
        controller = up.get();
        s = std::move(up);
    }

    ConnectInfo ci{
        .peer = peer,
        .oob_port = port,
    };
    s->connect(ci);
    MemRegion m = s->alloc_memory(REGION_BYTES, MemKind::CpuRam,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    PeerRef p = s->exchange_keys(m);
    ChannelDesc desc{
        .data_path = DataPath::CpuVerbs,
    };
    s->establish_channel(desc, m, p);

    std::uint64_t got = 0;
    void *outs[1] = {&got};
    std::uint64_t obytes[1] = {sizeof(got)};
    if (coproc) {
        coproc->set_coprocessor_fn(nullptr, nullptr); // built-in echo
        coproc->start();
        std::this_thread::sleep_for(std::chrono::seconds(3)); // serve ~3 s
        coproc->collect(outs, obytes, 1);
        coproc->stop();
    }
    else {
        // Controller: commit a work item, write the syndrome into
        // data_slot(), kick one round, then collect the correction.
        controller->commit_work_item(/*work_item_idx=*/0, /*in_bytes=*/sizeof(std::uint64_t),
                                     /*out_bytes=*/sizeof(std::uint64_t));
        controller->start();
        const std::uint64_t syndrome = DEMO_SYNDROME;
        std::memcpy(controller->data_slot(), &syndrome, sizeof(syndrome));
        controller->kick(0);
        controller->collect(outs, obytes, 1);
        controller->stop();
    }

    // Echo coprocessor: both roles observe the demo syndrome.
    const std::uint64_t expect = DEMO_SYNDROME;
    const bool pass = (got == expect);
    std::fprintf(stderr, "[%s] got=0x%llx expect=0x%llx -> %s\n", role.c_str(),
                 static_cast<unsigned long long>(got), static_cast<unsigned long long>(expect),
                 pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
