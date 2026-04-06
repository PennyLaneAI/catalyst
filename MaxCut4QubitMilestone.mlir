// Phase 1 Milestone — 4-qubit MaxCut DO-QAOA circuit
// Graph : 4-node cycle  (0-1-2-3-0)
// H     : -0.5*(Z0Z1 + Z1Z2 + Z2Z3 + Z3Z0)
// m     : 2 frozen hotspot qubits  →  2^2 = 4 sub-problems
// K     : 1 landscape cluster  (sparse cycle graph, s > sc ≈ 0.6)
// B_rep : 0.0000  (pure-ZZ Hamiltonian, no linear bias)
//
// Python decorator selected hotspot qubits: [0, 1]
// H_quad: #quantum.dense_graph<4, dense<[[0.000000e+00, -5.000000e-01, 0.000000e...
// H_lin : dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<4xf64>

module {

  // ----------------------------------------------------------------
  // Full DO-QAOA pipeline on the 4-qubit MaxCut circuit.
  // Input : %params_rep — optimised parameters for the representative
  //         sub-circuit (theta = [gamma_1, beta_1] at p=1).
  // Output: (implicit) %bitstring — best node assignment found.
  // ----------------------------------------------------------------
  func.func @maxcut_4qubit_doqaoa(%params_rep: !quantum.params) {

    // Step 1: Annotate frozen qubits and embed graph metadata.
    //   hotspot_count = m = 2
    //   hotspot_indices = [0, 1]  (highest degree-centrality nodes)
    //   h_quad = J_ij coupling matrix (MaxCut weights)
    //   h_lin  = h_i bias vector (all zero for pure-ZZ)
    %partition = quantum.freeze_partition {
        hotspot_count    = 2 : i32,
        hotspot_indices  = array<i32: 0, 1>,
        h_quad           = #quantum.dense_graph<4, dense<[[0.000000e+00, -5.000000e-01, 0.000000e+00, -5.000000e-01], [-5.000000e-01, 0.000000e+00, -5.000000e-01, 0.000000e+00], [0.000000e+00, -5.000000e-01, 0.000000e+00, -5.000000e-01], [-5.000000e-01, 0.000000e+00, -5.000000e-01, 0.000000e+00]]> : tensor<4x4xf64>>,
        h_lin            = dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<4xf64>
    } : !quantum.partition<4, 2>

    // Step 2: Cluster 2^m=4 sub-problem landscapes.
    //   K=1 because the 4-cycle is sparse (s > sc ≈ 0.6) — all
    //   sub-problems share the same landscape shape.
    %cluster_map = quantum.landscape_cluster(
        %partition : !quantum.partition<4, 2>)
        {k = 1 : i32} : !quantum.cluster_map<1>

    // Step 3: Pick the representative sub-circuit for the single cluster.
    %circuit_ref = quantum.select_representative(
        %cluster_map : !quantum.cluster_map<1>)
        : !quantum.circuit_ref

    // Step 4: Apply Bias-Aware Transfer Rule.
    //   |B_target - B_rep| = |0.0000 - 0.0000| = 0.0 < threshold=0.3
    //   → direct parameter copy, zero extra training sessions.
    %params_out = quantum.bias_transfer(
        %params_rep : !quantum.params)
        {B_rep = 0.000000e+00 : f64,
          B_target = 0.000000e+00 : f64,
          threshold = 3.000000e-01 : f64}
        : !quantum.params

    // Step 5: Select the sub-circuit bitstring with minimum <H>.
    %bitstring = quantum.aggregate_min(
        %params_out : !quantum.params)
        : !quantum.bitstring

    func.return
  }

}
