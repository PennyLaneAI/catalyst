// Copyright 2026 Haiqu, Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --split-input-file --verify-diagnostics

func.func private @circuit(%arg0: tensor<f64>) -> tensor<4xf64>

func.func @remValid(%x: tensor<f64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    %out:3 = mitigation.rem @circuit(%x) runCalibration(true) {resultSegmentSizes = array<i32: 1, 1, 1>} : (tensor<f64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    %res = mitigation.rem @circuit(%x) runCalibration(false) {resultSegmentSizes = array<i32: 1, 0, 0>} : (tensor<f64>) -> tensor<4xf64>
    func.return %out#0, %out#1, %out#2, %res : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// -----

func.func @remMissingCallee() -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    // expected-error@+1 {{invalid function name specified: @doesNotExist}}
    %out:3 = mitigation.rem @doesNotExist() runCalibration(true) {resultSegmentSizes = array<i32: 1, 1, 1>} : () -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// -----

func.func private @circuit(%arg0: tensor<f64>) -> tensor<4xf64>

func.func @remArgumentMismatch() -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    // expected-error@+1 {{argument types do not match callee @circuit: expected ('tensor<f64>'), got ()}}
    %out:3 = mitigation.rem @circuit() runCalibration(true) {resultSegmentSizes = array<i32: 1, 1, 1>} : () -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// -----

func.func private @circuit() -> tensor<4xf64>

func.func @remCalleeResultMismatch() -> (tensor<8xf64>, tensor<4xf64>, tensor<4xf64>) {
    // expected-error@+1 {{callee_result types do not match the results of callee @circuit: expected ('tensor<4xf64>'), got ('tensor<8xf64>')}}
    %out:3 = mitigation.rem @circuit() runCalibration(true) {resultSegmentSizes = array<i32: 1, 1, 1>} : () -> (tensor<8xf64>, tensor<4xf64>, tensor<4xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<8xf64>, tensor<4xf64>, tensor<4xf64>
}

// -----

func.func private @circuit() -> tensor<4xf64>

func.func @remNoCalibrationWithCalibrationResults() -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    // expected-error@+1 {{expected no zeros and ones results when runCalibration is false, got 1 zeros and 1 ones results}}
    %out:3 = mitigation.rem @circuit() runCalibration(false) {resultSegmentSizes = array<i32: 1, 1, 1>} : () -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// -----

func.func private @circuit() -> tensor<4xf64>

func.func @remCalibrationWithoutCalibrationResults() -> tensor<4xf64> {
    // expected-error@+1 {{expected one zeros and one ones result when runCalibration is true, got 0 zeros and 0 ones results}}
    %out = mitigation.rem @circuit() runCalibration(true) {resultSegmentSizes = array<i32: 1, 0, 0>} : () -> tensor<4xf64>
    func.return %out : tensor<4xf64>
}

// -----

func.func private @circuit() -> tensor<4xf64>

func.func @remCalibrationTypeMismatch() -> (tensor<4xf64>, tensor<4xf64>, tensor<8xf64>) {
    // expected-error@+1 {{expected zeros and ones results of the same type, got 'tensor<4xf64>' and 'tensor<8xf64>'}}
    %out:3 = mitigation.rem @circuit() runCalibration(true) {resultSegmentSizes = array<i32: 1, 1, 1>} : () -> (tensor<4xf64>, tensor<4xf64>, tensor<8xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<4xf64>, tensor<4xf64>, tensor<8xf64>
}
