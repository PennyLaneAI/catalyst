module @module {
  func.func private @bar() -> (tensor<i64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    return %c : tensor<i64>
  }
}

