func.func @my_circuit(%q0 : !oq.qubit) {
    %phi = arith.constant 0.3 : f64

    oq.RZ(%phi) %q0 : !oq.qubit

    func.return
}
