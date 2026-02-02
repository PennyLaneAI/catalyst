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

// CHECK: llvm.func @now_mu() -> i64
// CHECK: llvm.func @at_mu(i64)
// CHECK: llvm.func @rtio_get_counter() -> i64
// CHECK: llvm.func @rtio_init()
// CHECK: llvm.func @delay_mu(i64)
// CHECK: llvm.func internal @__rtio_set_frequency(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: f64)
// CHECK: llvm.func @rtio_output(i32, i32)
// CHECK: llvm.func internal @__rtio_config_spi(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32)
// CHECK: llvm.func internal fastcc @__rtio_sec_to_mu(%arg0: f64) -> i64

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

    // Test sequential pulse on same channel
    // CHECK: llvm.call tail @at_mu
    // CHECK: llvm.call fastcc tail @__rtio_sec_to_mu
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
    // CHECK: llvm.call fastcc tail @__rtio_sec_to_mu
    // CHECK: llvm.call tail @rtio_output
    // CHECK: llvm.call fastcc tail @delay_mu
    // CHECK: llvm.call tail @rtio_output
    %1 = rtio.pulse %ch0 duration(%cst_dur) frequency(%cst_freq) phase(%cst_phase) wait(%0) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event

    // Second pulse, sequential, waits for first
    // CHECK: llvm.call tail @at_mu
    // CHECK: llvm.call fastcc tail @__rtio_sec_to_mu
    // CHECK: llvm.call tail @rtio_output
    // CHECK: llvm.call fastcc tail @delay_mu
    // CHECK: llvm.call tail @rtio_output
    %2 = rtio.pulse %ch0 duration(%cst_dur) frequency(%cst_freq) phase(%cst_phase) wait(%1) {offset = 0 : i64} : <"dds", [2 : i64], 0> -> !rtio.event

    // CHECK: return
    return
  }
}

