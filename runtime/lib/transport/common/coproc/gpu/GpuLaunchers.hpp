// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "Transport.hpp" // CoprocLaunchDesc

namespace catalyst::transport::coproc {

// CoprocessorLauncherFn launchers: each enqueues the persistent detect ->
// decode -> write-handoff kernel (one per decoder) on the session's stream.
// `desc` is the datapath to wire the kernel to; `ctx` is unused. Return 0 if
// the launch succeeded, nonzero if it failed.
extern "C" int gpu_echo_launcher(const CoprocLaunchDesc *desc, void *ctx);
extern "C" int gpu_steane_launcher(const CoprocLaunchDesc *desc, void *ctx);

// Built-in launcher the session uses when no launcher is bound
// (coproc_launcher_ == nullptr); identical to the echo launcher.
int default_echo_launcher(const CoprocLaunchDesc *desc, void *ctx);

} // namespace catalyst::transport::coproc
