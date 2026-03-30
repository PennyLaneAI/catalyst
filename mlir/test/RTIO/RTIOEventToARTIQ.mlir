// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --convert-rtio-event-to-artiq --split-input-file | FileCheck %s

// CHECK-DAG: llvm.func @now_mu() -> i64
// CHECK-DAG: llvm.func @rtio_init()
// CHECK-DAG: llvm.func @at_mu(i64)
// CHECK-DAG: llvm.func @rtio_get_counter() -> i64
// CHECK-DAG: llvm.func @delay_mu(i64)
// CHECK-DAG: llvm.func internal @__rtio_set_frequency(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: f64)
// CHECK-DAG: llvm.func @rtio_output(i32, i32)
// CHECK-DAG: llvm.func internal @__rtio_config_spi(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32)
// CHECK-DAG: llvm.func internal fastcc @__rtio_sec_to_mu(%arg0: f64) -> i64

// CHECK-LABEL: func.func @__kernel__()
// CHECK-SAME: attributes {diff_method = "parameter-shift", qnode}
module @circuit attributes {rtio.config = #rtio.config<{core_addr = "172.31.9.64", device_db = {core = {arguments = {analyzer_proxy = "core_analyzer", host = "172.31.9.64", ref_period = 1.000000e-09 : f64, satellite_cpu_targets = {"1" = "rv32g"}, target = "cortexa9"}, class = "Core", module = "artiq.coredevice.core", type = "local"}, spi_urukul0 = {arguments = {channel = 17 : i64}, class = "SPIMaster", module = "artiq.coredevice.spi2", type = "local"}, ttl_urukul0_io_update = {arguments = {channel = 18 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw0 = {arguments = {channel = 19 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw1 = {arguments = {channel = 20 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw2 = {arguments = {channel = 21 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw3 = {arguments = {channel = 22 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, urukul0_ch0 = {arguments = {chip_select = 4 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw0"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_ch1 = {arguments = {chip_select = 5 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw1"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_ch2 = {arguments = {chip_select = 6 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw2"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_ch3 = {arguments = {chip_select = 7 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw3"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_cpld = {arguments = {clk_div = 0 : i64, clk_sel = 2 : i64, io_update_device = "ttl_urukul0_io_update", refclk = 125000000 : i64, spi_device = "spi_urukul0", sync_device}, class = "CPLD", module = "artiq.coredevice.urukul", type = "local"}}}>} {
  memref.global "private" constant @__qubit_map_0 : memref<2xindex> = dense<[0, 1]>
  func.func @__kernel__() attributes {diff_method = "parameter-shift", qnode} {
    %cst = arith.constant 1.1618250000000001E-6 : f64
    %cst_0 = arith.constant 8.298750e-06 : f64
    %cst_1 = arith.constant 1.6597500000000003E-7 : f64
    %cst_2 = arith.constant 19100000.100724373 : f64
    %cst_3 = arith.constant 20900000.012723904 : f64
    %cst_4 = arith.constant 19000000.026035815 : f64
    %cst_5 = arith.constant 21000000.087412462 : f64
    %cst_6 = arith.constant 19999999.977146666 : f64
    %cst_7 = arith.constant 0.000000e+00 : f64

    // Test rtio.empty,  should initialize RTIO and return a timestamp
    // CHECK: llvm.call fastcc tail @rtio_init()
    // CHECK: %[[COUNTER:.*]] = llvm.call fastcc tail @rtio_get_counter()
    // CHECK: %[[OFFSET:.*]] = arith.constant 125000 : i64
    // CHECK: %[[INIT_TIME:.*]] = arith.addi %[[COUNTER]], %[[OFFSET]]
    // CHECK: llvm.call tail @at_mu(%[[INIT_TIME]])
    %0 = rtio.empty : !rtio.event

    // Test rtio.channel, creates channel reference
    %1 = rtio.channel : !rtio.channel<"dds", [2 : i64], 2>
    %3 = rtio.channel : !rtio.channel<"dds", [2 : i64], 0>

    // Test rtio.pulse with wait on empty, should set frequency and generate TTL pulse
    // First pulse on channel 2 waiting on empty event
    // CHECK: llvm.call tail @now_mu()
    // CHECK: llvm.call tail @at_mu
    // CHECK: llvm.call @__rtio_set_frequency
    // CHECK: llvm.call tail @now_mu()
    %2 = rtio.pulse %1 duration(%cst_1) frequency(%cst_6) phase(%cst_7) wait(%0) {offset = 0 : i64} : <"dds", [2 : i64], 2> -> !rtio.event

    // Test parallel pulses, both wait on same event
    // CHECK: llvm.call tail @at_mu
    // CHECK: llvm.call @__rtio_set_frequency
    %4 = rtio.pulse %3 duration(%cst_1) frequency(%cst_6) phase(%cst_7) wait(%0) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event

    // Test sequential pulse on same channel (duration via max(duration_mu, minTTL), not __rtio_sec_to_mu)
    // CHECK: llvm.call tail @at_mu
    // CHECK: arith.maxsi
    // CHECK: llvm.call tail @rtio_output
    // CHECK: llvm.call fastcc tail @delay_mu
    // CHECK: llvm.call tail @rtio_output
    %5 = rtio.pulse %3 duration(%cst_1) frequency(%cst_6) phase(%cst_7) wait(%4) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event

    // Test rtio.sync, synchronizes multiple events using maxsi
    // CHECK: arith.maxsi
    // CHECK: llvm.call tail @at_mu
    %6 = rtio.sync %5, %2 : !rtio.event

    // Test multiple parallel pulses after sync
    %7 = rtio.pulse %3 duration(%cst_0) frequency(%cst_5) phase(%cst_7) wait(%6) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event
    %8 = rtio.channel : !rtio.channel<"dds", [2 : i64], 1>
    %9 = rtio.pulse %8 duration(%cst_0) frequency(%cst_4) phase(%cst_7) wait(%6) {offset = 1 : i64} : <"dds", [2 : i64], 1> -> !rtio.event
    %10 = rtio.pulse %1 duration(%cst_0) frequency(%cst_3) phase(%cst_7) wait(%6) {offset = 0 : i64} : <"dds", [2 : i64], 2> -> !rtio.event
    %11 = rtio.channel : !rtio.channel<"dds", [2 : i64], 3>
    %12 = rtio.pulse %11 duration(%cst_0) frequency(%cst_2) phase(%cst_7) wait(%6) {offset = 1 : i64} : <"dds", [2 : i64], 3> -> !rtio.event

    // Test sync with 4 events
    // CHECK: arith.maxsi
    // CHECK: arith.maxsi
    // CHECK: arith.maxsi
    // CHECK: llvm.call tail @at_mu
    %13 = rtio.sync %7, %9, %10, %12 : !rtio.event

    // Final pulses after sync
    %14 = rtio.pulse %3 duration(%cst) frequency(%cst_6) phase(%cst_7) wait(%13) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event
    %15 = rtio.pulse %1 duration(%cst) frequency(%cst_6) phase(%cst_7) wait(%13) {offset = 0 : i64} : <"dds", [2 : i64], 2> -> !rtio.event
    %16 = rtio.pulse %3 duration(%cst) frequency(%cst_6) phase(%cst_7) wait(%14) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event

    // CHECK: return
    return
  }
}

// -----

// CHECK-LABEL: func.func @__kernel__()
module @simple_sequential attributes {rtio.config = #rtio.config<{core_addr = "172.31.9.64", device_db = {core = {arguments = {host = "172.31.9.64", ref_period = 1.000000e-09 : f64, target = "cortexa9"}, class = "Core", module = "artiq.coredevice.core", type = "local"}, spi_urukul0 = {arguments = {channel = 17 : i64}, class = "SPIMaster", module = "artiq.coredevice.spi2", type = "local"}, ttl_urukul0_io_update = {arguments = {channel = 18 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw0 = {arguments = {channel = 19 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, urukul0_ch0 = {arguments = {chip_select = 4 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw0"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_cpld = {arguments = {clk_div = 0 : i64, clk_sel = 2 : i64, io_update_device = "ttl_urukul0_io_update", refclk = 125000000 : i64, spi_device = "spi_urukul0", sync_device}, class = "CPLD", module = "artiq.coredevice.urukul", type = "local"}}}>} {
  memref.global "private" constant @__qubit_map_0 : memref<1xindex> = dense<0>
  func.func @__kernel__() attributes {diff_method = "parameter-shift", qnode} {
    %cst_dur = arith.constant 1.0e-6 : f64
    %cst_freq = arith.constant 20000000.0 : f64
    %cst_phase = arith.constant 0.0 : f64

    // CHECK: llvm.call fastcc tail @rtio_init()
    %0 = rtio.empty : !rtio.event

    %ch0 = rtio.channel : !rtio.channel<"dds", [2 : i64], 0>

    // First pulse, sets frequency and generates TTL
    // CHECK: llvm.call @__rtio_set_frequency
    // CHECK: llvm.call fastcc tail @delay_mu
    // CHECK: arith.maxsi
    // CHECK: llvm.call tail @rtio_output
    // CHECK: llvm.call fastcc tail @delay_mu
    // CHECK: llvm.call tail @rtio_output
    %1 = rtio.pulse %ch0 duration(%cst_dur) frequency(%cst_freq) phase(%cst_phase) wait(%0) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event

    // Second pulse, sequential, waits for first
    // CHECK: llvm.call tail @at_mu
    // CHECK: arith.maxsi
    // CHECK: llvm.call tail @rtio_output
    // CHECK: llvm.call fastcc tail @delay_mu
    // CHECK: llvm.call tail @rtio_output
    %2 = rtio.pulse %ch0 duration(%cst_dur) frequency(%cst_freq) phase(%cst_phase) wait(%1) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event

    // CHECK: return
    return
  }
}

// -----

// Synchronous RPC with no arguments (program_awg, awg_close).
// CHECK-DAG: llvm.func @rpc_send(i32, !llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @rpc_recv(!llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global private constant @__rtio_str__n
module @rpc_no_args attributes {rtio.config = #rtio.config<{core_addr = "172.31.9.64", device_db = {core = {arguments = {host = "172.31.9.64", ref_period = 1.000000e-09 : f64, target = "cortexa9"}, class = "Core", module = "artiq.coredevice.core", type = "local"}, spi_urukul0 = {arguments = {channel = 17 : i64}, class = "SPIMaster", module = "artiq.coredevice.spi2", type = "local"}, ttl_urukul0_io_update = {arguments = {channel = 18 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw0 = {arguments = {channel = 19 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, urukul0_ch0 = {arguments = {chip_select = 4 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw0"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_cpld = {arguments = {clk_div = 0 : i64, clk_sel = 2 : i64, io_update_device = "ttl_urukul0_io_update", refclk = 125000000 : i64, spi_device = "spi_urukul0", sync_device}, class = "CPLD", module = "artiq.coredevice.urukul", type = "local"}}}>} {

  // CHECK-LABEL: func.func @__kernel__()
  func.func @__kernel__() {

    // CHECK: llvm.call @rpc_send({{.*}}) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    // CHECK: %[[SZ0:.*]] = llvm.call @rpc_recv({{.*}}) : (!llvm.ptr) -> i32
    // CHECK: scf.while
    rtio.rpc @program_awg

    // CHECK: llvm.call @rpc_send({{.*}}) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @rpc_recv({{.*}}) : (!llvm.ptr) -> i32
    rtio.rpc @awg_close
    return
  }
}

// -----

// Async RPC with args (i64, i64, f64) -> void.
// CHECK-DAG: llvm.func @rpc_send_async(i32, !llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.mlir.global private constant @__rtio_str_IIf_n
module @rpc_async_with_args attributes {rtio.config = #rtio.config<{core_addr = "172.31.9.64", device_db = {core = {arguments = {host = "172.31.9.64", ref_period = 1.000000e-09 : f64, target = "cortexa9"}, class = "Core", module = "artiq.coredevice.core", type = "local"}, spi_urukul0 = {arguments = {channel = 17 : i64}, class = "SPIMaster", module = "artiq.coredevice.spi2", type = "local"}, ttl_urukul0_io_update = {arguments = {channel = 18 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw0 = {arguments = {channel = 19 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, urukul0_ch0 = {arguments = {chip_select = 4 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw0"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_cpld = {arguments = {clk_div = 0 : i64, clk_sel = 2 : i64, io_update_device = "ttl_urukul0_io_update", refclk = 125000000 : i64, spi_device = "spi_urukul0", sync_device}, class = "CPLD", module = "artiq.coredevice.urukul", type = "local"}}}>} {
  // CHECK-LABEL: func.func @__kernel__
  func.func @__kernel__(%key: i64, %idx: i64, %val: f64) {
    // CHECK: llvm.call @rpc_send_async({{.*}}) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    // CHECK-NOT: llvm.call @rpc_recv
    rtio.rpc @transfer_data async (%key, %idx, %val : i64, i64, f64)
    return
  }
}

// -----

// ID deduplication: set_dataset called twice gets the same rpc_id.
module @rpc_id_deduplication attributes {rtio.config = #rtio.config<{core_addr = "172.31.9.64", device_db = {core = {arguments = {host = "172.31.9.64", ref_period = 1.000000e-09 : f64, target = "cortexa9"}, class = "Core", module = "artiq.coredevice.core", type = "local"}, spi_urukul0 = {arguments = {channel = 17 : i64}, class = "SPIMaster", module = "artiq.coredevice.spi2", type = "local"}, ttl_urukul0_io_update = {arguments = {channel = 18 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw0 = {arguments = {channel = 19 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, urukul0_ch0 = {arguments = {chip_select = 4 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw0"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_cpld = {arguments = {clk_div = 0 : i64, clk_sel = 2 : i64, io_update_device = "ttl_urukul0_io_update", refclk = 125000000 : i64, spi_device = "spi_urukul0", sync_device}, class = "CPLD", module = "artiq.coredevice.urukul", type = "local"}}}>} {
  // CHECK-LABEL: func.func @__kernel__
  func.func @__kernel__(%key: i64, %val: i64) {
    // set_dataset -> rpc_id = 1

    // CHECK: arith.constant 1 : i32
    // CHECK: llvm.call @rpc_send({{.*}}) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @rpc_recv({{.*}}) : (!llvm.ptr) -> i32
    // CHECK: arith.constant 1 : i32
    // CHECK: llvm.call @rpc_send({{.*}}) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @rpc_recv({{.*}}) : (!llvm.ptr) -> i32
    rtio.rpc @set_dataset (%key, %val : i64, i64)
    rtio.rpc @set_dataset (%key, %val : i64, i64)

    // program_awg -> rpc_id = 2

    // CHECK: arith.constant 2 : i32
    // CHECK: llvm.call @rpc_send({{.*}}) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @rpc_recv({{.*}}) : (!llvm.ptr) -> i32
    rtio.rpc @program_awg
    return
  }
}

// -----

// Sync RPC with return value (i64 return, no args).
// CHECK-DAG: llvm.mlir.global private constant @__rtio_str__I
module @rpc_with_return attributes {rtio.config = #rtio.config<{core_addr = "172.31.9.64", device_db = {core = {arguments = {host = "172.31.9.64", ref_period = 1.000000e-09 : f64, target = "cortexa9"}, class = "Core", module = "artiq.coredevice.core", type = "local"}, spi_urukul0 = {arguments = {channel = 17 : i64}, class = "SPIMaster", module = "artiq.coredevice.spi2", type = "local"}, ttl_urukul0_io_update = {arguments = {channel = 18 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl_urukul0_sw0 = {arguments = {channel = 19 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, urukul0_ch0 = {arguments = {chip_select = 4 : i64, cpld_device = "urukul0_cpld", pll_en = 1 : i64, pll_n = 32 : i64, sw_device = "ttl_urukul0_sw0"}, class = "AD9910", module = "artiq.coredevice.ad9910", type = "local"}, urukul0_cpld = {arguments = {clk_div = 0 : i64, clk_sel = 2 : i64, io_update_device = "ttl_urukul0_io_update", refclk = 125000000 : i64, spi_device = "spi_urukul0", sync_device}, class = "CPLD", module = "artiq.coredevice.urukul", type = "local"}}}>} {
  // CHECK-LABEL: func.func @__kernel__
  func.func @__kernel__() -> i64 {
    // CHECK: llvm.call @rpc_send({{.*}}) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    // CHECK: %[[SZ:.*]] = llvm.call @rpc_recv(%[[BUF:.*]]) : (!llvm.ptr) -> i32
    // CHECK: scf.while
    // CHECK: llvm.load %[[BUF]] : !llvm.ptr -> i64
    %x = rtio.rpc @get_counter -> i64
    return %x : i64
  }
}

// -----

// RTIO measurement
// CHECK-LABEL: func.func @__kernel__()
// CHECK-SAME: attributes {qnode}
module @measure_rtio_to_artiq attributes {rtio.config = #rtio.config<{device_db = {core = {arguments = {host = "172.31.9.64", ref_period = 1.000000e-09 : f64, target = "cortexa9"}, class = "Core", module = "artiq.coredevice.core", type = "local"}, ttl0 = {arguments = {channel = 5 : i64, gate_latency_mu = 104 : i64}, class = "TTLInOut", module = "artiq.coredevice.ttl", type = "local"}, ttl6 = {arguments = {channel = 11 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}, ttl7 = {arguments = {channel = 12 : i64}, class = "TTLOut", module = "artiq.coredevice.ttl", type = "local"}}}>} {
  memref.global "private" constant @__qubit_map_0 : memref<1xindex> = dense<0>
  func.func @__kernel__() attributes {qnode} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e-04 : f64
    %0 = rtio.empty : !rtio.event
    %1 = rtio.channel : !rtio.channel<"ttl", [1 : i64], 0>
    %2 = rtio.pulse %1 duration(%cst_0) frequency(%cst) phase(%cst) wait(%0) {_measurement, offset = 0 : i64} : <"ttl", [1 : i64], 0> -> !rtio.event
    %3 = rtio.readout %2 : !rtio.event -> i32
    return
  }
  func.func private @__rtio_init_dataset() {
    rtio.rpc @init_dataset rpc_id(1) async
    return
  }
  func.func private @__rtio_transfer_measurement_results(%arg0: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %c1 step %c1_0 {
      %0 = arith.index_cast %arg1 : index to i32
      %1 = memref.load %arg0[%arg1] : memref<1xi32>
      rtio.rpc @transfer_measurement_result rpc_id(2) async(%0, %1 : i32, i32)
    }
    return
  }
}

// CHECK: memref.alloca{{.*}} : memref<1xi32>
// CHECK: call @__rtio_init_dataset
// CHECK: llvm.call @__rtio_count
// CHECK: memref.store
// CHECK: call @__rtio_transfer_measurement_results
// CHECK: llvm.call{{.*}} @__rtio_wait_until_mu
