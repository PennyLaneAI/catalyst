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

// RUN: quantum-opt %s | quantum-opt | FileCheck %s

// Smoke test for the transport dialect

// CHECK-LABEL: func.func @transport_smoketest
func.func @transport_smoketest(%payload: memref<?xi8>, %reply: memref<?xi8>) {
  // CHECK: transport.create {{.*}} -> !transport.session<controller>
  %ct = transport.create {backend_lib = "libbackend.so", config = "cfg"} -> !transport.session<controller>
  // CHECK: transport.create {{.*}} -> !transport.session<coprocessor>
  %co = transport.create {backend_lib = "libbackend.so", config = "cfg"} -> !transport.session<coprocessor>

  // CHECK: transport.connect_async %{{.*}} : !transport.session<coprocessor> -> !transport.token
  %t1 = transport.connect_async %co {peer = "127.0.0.1", oob_port = 18590 : ui16} : !transport.session<coprocessor> -> !transport.token
  // CHECK: transport.connect %{{.*}} : !transport.session<controller>
  transport.connect %ct {peer = "127.0.0.1", oob_port = 18590 : ui16} : !transport.session<controller>
  // CHECK: transport.await %{{.*}} : !transport.token
  transport.await %t1 : !transport.token
  // CHECK: transport.exchange_keys_async %{{.*}} : !transport.session<coprocessor> -> !transport.token
  %t2 = transport.exchange_keys_async %co : !transport.session<coprocessor> -> !transport.token
  transport.await %t2 : !transport.token
  // CHECK: transport.exchange_keys %{{.*}} : !transport.session<controller>
  transport.exchange_keys %ct : !transport.session<controller>

  // CHECK: transport.establish_channel %{{.*}} "rdma" : !transport.session<controller>
  transport.establish_channel %ct "rdma" : !transport.session<controller>
  // CHECK: transport.establish_channel %{{.*}} "memcpy" : !transport.session<coprocessor>
  transport.establish_channel %co "memcpy" : !transport.session<coprocessor>

  // CHECK: transport.set_coprocessor_fn %{{.*}} {symbol = "foo"} : !transport.session<coprocessor>
  transport.set_coprocessor_fn %co {symbol = "foo"} : !transport.session<coprocessor>
  // CHECK: transport.set_message_sizes %{{.*}} : !transport.session<controller>
  transport.set_message_sizes %ct {in_bytes = 8 : i64, out_bytes = 8 : i64} : !transport.session<controller>

  transport.start %co : !transport.session<coprocessor>
  transport.start %ct : !transport.session<controller>

  // CHECK: transport.get_session : !transport.session<controller>
  %ct2 = transport.get_session : !transport.session<controller>

  // CHECK: transport.stage_payload %{{.*}}, %{{.*}} : !transport.session<controller>, memref<?xi8>
  transport.stage_payload %ct2, %payload : !transport.session<controller>, memref<?xi8>
  // CHECK: transport.post %{{.*}} : !transport.session<controller>
  transport.post %ct2 : !transport.session<controller>
  // CHECK: transport.collect %{{.*}}, %{{.*}} : !transport.session<controller>, memref<?xi8>
  transport.collect %ct2, %reply : !transport.session<controller>, memref<?xi8>
  // CHECK: transport.last_rtt_ns %{{.*}} : !transport.session<controller> -> i64
  %rtt = transport.last_rtt_ns %ct2 : !transport.session<controller> -> i64

  transport.stop %ct : !transport.session<controller>
  transport.destroy %ct : !transport.session<controller>
  transport.destroy %co : !transport.session<coprocessor>
  return
}
