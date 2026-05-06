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

// WarmStartSchedulerPass — Phase 3, Task 5
//
// Runs a compile-time Adam optimisation loop for every mode-2 (warm-start)
// sub-problem using the existing EnergyEval infrastructure.
//
// Algorithm per mode-2 sub-problem k:
//   1. Load init [γ₀, β₀] from init_params[buffer_slot_map[k]]
//      (basin / shortcut angles of the cluster representative).
//   2. Adam loop (up to warmstartEpochs steps):
//        a. Compute gradient via central finite differences with step h.
//        b. Update Adam moments m, v (bias-corrected).
//        c. Apply parameter update: θ -= lr × m̂ / (√v̂ + ε).
//        d. Check convergence: ||∇θ||₂ < gradNormTol → stop early.
//   3. Record final (γ, β), convergence flag, epoch count, final ⟨H_k⟩.
//
// Output tensors stored on freeze_partition:
//   warmstart_params       tensor<2^m × 2 × f64>
//   warmstart_converged    array<i32, 2^m>   (1=converged, 0=limit, -1=N/A)
//   warmstart_epochs_used  array<i32, 2^m>
//   warmstart_final_energy tensor<2^m × f64>
//
// Requires: init_params, buffer_slot_map (doqaoa-shared-buffer),
//           transfer_modes (doqaoa-representative-selection),
//           cluster_assignments, h_quad, h_lin, hotspot_indices.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/EnergyEval.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;
using namespace catalyst::quantum::energy;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_WARMSTARTSCHEDULERPASS
#define GEN_PASS_DEF_WARMSTARTSCHEDULERPASS
#include "Quantum/Transforms/Passes.h.inc"

// ─────────────────────────────────────────────────────────────────────────────
// Graph extraction helpers (mirrors BiasShiftAnalysis)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> wsExtractJ(Operation *op, unsigned &numNodes)
{
    auto hq = op->getAttr("h_quad");
    if (!hq) {
        numNodes = 0;
        return {};
    }

    if (auto dense = dyn_cast<DenseGraphAttr>(hq)) {
        numNodes = static_cast<unsigned>(dense.getNumNodes());
        auto data = dense.getWeights();
        std::vector<double> J;
        J.reserve(data.size());
        for (double v : data.getValues<double>())
            J.push_back(v);
        return J;
    }
    if (auto sparse = dyn_cast<SparseGraphAttr>(hq)) {
        numNodes = static_cast<unsigned>(sparse.getNumNodes());
        unsigned N = numNodes;
        std::vector<double> J(static_cast<size_t>(N) * N, 0.0);
        auto rows = sparse.getRowIndices();
        auto cols = sparse.getColIndices();
        auto weights = sparse.getWeights();
        auto wt = weights.getValues<double>();
        auto wit = wt.begin();
        for (unsigned e = 0; e < static_cast<unsigned>(rows.size()); ++e, ++wit) {
            unsigned r = static_cast<unsigned>(rows[e]);
            unsigned c = static_cast<unsigned>(cols[e]);
            J[r * N + c] = *wit;
            J[c * N + r] = *wit;
        }
        return J;
    }
    numNodes = 0;
    return {};
}

static std::vector<double> wsExtractH(Operation *op, unsigned numNodes)
{
    std::vector<double> h(numNodes, 0.0);
    auto hl = op->getAttr("h_lin");
    if (!hl)
        return h;
    auto dens = cast<DenseElementsAttr>(hl);
    unsigned i = 0;
    for (double v : dens.getValues<double>()) {
        if (i >= numNodes)
            break;
        h[i++] = v;
    }
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam one-step update (in-place on gamma/beta)
// ─────────────────────────────────────────────────────────────────────────────

struct AdamState {
    double mg = 0, mb = 0; // first-moment estimates for γ, β
    double vg = 0, vb = 0; // second-moment estimates
    unsigned t = 0;        // step counter (1-based)
};

static void adamStep(double &gamma, double &beta, double grad_g, double grad_b, AdamState &st,
                     double lr, double beta1, double beta2, double eps)
{
    ++st.t;
    st.mg = beta1 * st.mg + (1.0 - beta1) * grad_g;
    st.mb = beta1 * st.mb + (1.0 - beta1) * grad_b;
    st.vg = beta2 * st.vg + (1.0 - beta2) * grad_g * grad_g;
    st.vb = beta2 * st.vb + (1.0 - beta2) * grad_b * grad_b;

    double mhat_g = st.mg / (1.0 - std::pow(beta1, static_cast<double>(st.t)));
    double mhat_b = st.mb / (1.0 - std::pow(beta1, static_cast<double>(st.t)));
    double vhat_g = st.vg / (1.0 - std::pow(beta2, static_cast<double>(st.t)));
    double vhat_b = st.vb / (1.0 - std::pow(beta2, static_cast<double>(st.t)));

    gamma -= lr * mhat_g / (std::sqrt(vhat_g) + eps);
    beta -= lr * mhat_b / (std::sqrt(vhat_b) + eps);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass implementation
// ─────────────────────────────────────────────────────────────────────────────

struct WarmStartSchedulerPass : impl::WarmStartSchedulerPassBase<WarmStartSchedulerPass> {
    using WarmStartSchedulerPassBase::WarmStartSchedulerPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();
        Builder builder(ctx);

        unsigned maxEp = warmstartEpochs;
        double lr = learningRate;
        double tol = gradNormTol;
        double h = finiteDiffStep;

        // Adam hypers (fixed, standard values)
        constexpr double beta1 = 0.9;
        constexpr double beta2 = 0.999;
        constexpr double eps = 1e-8;

        func.walk(
            [&](FreezePartitionOp op) {
                // ── Guard: need shared-buffer + representative-selection attrs ─
                auto initParamsAttr = op->getAttr("init_params");
                auto slotMapAttr = op->getAttr("buffer_slot_map");
                auto modesAttr = op->getAttr("transfer_modes");

                if (!initParamsAttr || !slotMapAttr || !modesAttr) {
                    op->emitWarning() << "doqaoa-warmstart-scheduler: missing shared-buffer or "
                                         "representative-selection attributes; run "
                                         "doqaoa-shared-buffer and doqaoa-representative-selection "
                                         "first, skipping";
                    return;
                }

                // ── Extract dimensions ────────────────────────────────────────
                auto hotspotAttr = op.getHotspotIndices();
                unsigned m = static_cast<unsigned>(hotspotAttr.size());
                unsigned numSP = 1u << m;

                // ── Build graph descriptor for EnergyEval ─────────────────────
                unsigned numNodes = 0;
                std::vector<double> J = wsExtractJ(op, numNodes);
                std::vector<double> hVec = wsExtractH(op, numNodes);
                if (numNodes == 0) {
                    op->emitWarning()
                        << "doqaoa-warmstart-scheduler: missing h_quad attribute, skipping";
                    return;
                }

                std::vector<int32_t> hotspotVec;
                for (auto idx : hotspotAttr)
                    hotspotVec.push_back(static_cast<int32_t>(idx));

                GraphDesc graph;
                graph.numNodes = numNodes;
                graph.J = std::move(J);
                graph.h = std::move(hVec);
                graph.hotspotIndices = std::move(hotspotVec);

                // ── Extract init_params (shape K × 2) ─────────────────────────
                auto initDense = cast<DenseElementsAttr>(initParamsAttr);
                std::vector<double> initFlat;
                for (double v : initDense.getValues<double>())
                    initFlat.push_back(v);
                // initFlat[c*2+0] = gamma_init for cluster c
                // initFlat[c*2+1] = beta_init  for cluster c

                auto slotArr = cast<DenseI32ArrayAttr>(slotMapAttr);
                auto modesArr = cast<DenseI32ArrayAttr>(modesAttr);

                // ── Per-sub-problem Adam warm-start ───────────────────────────
                // Output arrays (original k order)
                std::vector<double> finalParams(numSP * 2, 0.0);
                std::vector<int32_t> converged(numSP, -1); // -1 = N/A
                std::vector<int32_t> epochsUsed(numSP, 0);
                std::vector<double> finalEnergy(numSP, std::numeric_limits<double>::quiet_NaN());

                int32_t warmCount = 0, convCount = 0;

                for (unsigned k = 0; k < numSP; ++k) {
                    int32_t mode = (k < static_cast<unsigned>(modesArr.size())) ? modesArr[k] : 1;
                    int32_t slot = (k < static_cast<unsigned>(slotArr.size())) ? slotArr[k] : 0;

                    // Slot index into initFlat (row = slot, cols = [gamma, beta])
                    double g0 = initFlat[static_cast<unsigned>(slot) * 2 + 0];
                    double b0 = initFlat[static_cast<unsigned>(slot) * 2 + 1];

                    if (mode == 0) {
                        // Representative: store init (runtime refines in phase 1)
                        finalParams[k * 2 + 0] = g0;
                        finalParams[k * 2 + 1] = b0;
                        converged[k] = -1;
                        epochsUsed[k] = 0;
                        // finalEnergy stays NaN — not yet evaluated
                        continue;
                    }

                    if (mode == 1) {
                        // Direct copy: store rep's init unchanged
                        finalParams[k * 2 + 0] = g0;
                        finalParams[k * 2 + 1] = b0;
                        converged[k] = -1;
                        epochsUsed[k] = 0;
                        continue;
                    }

                    // mode == 2: warm-start Adam loop
                    ++warmCount;
                    double gk = g0, bk = b0;
                    AdamState st;
                    bool conv = false;
                    unsigned ep = 0;

                    for (ep = 1; ep <= maxEp; ++ep) {
                        // Central finite-difference gradient
                        double Ep = evaluateEnergy(graph, k, gk, bk);
                        double Epg = evaluateEnergy(graph, k, gk + h, bk);
                        double Emg = evaluateEnergy(graph, k, gk - h, bk);
                        double Epb = evaluateEnergy(graph, k, gk, bk + h);
                        double Emb = evaluateEnergy(graph, k, gk, bk - h);

                        double grad_g = (Epg - Emg) / (2.0 * h);
                        double grad_b = (Epb - Emb) / (2.0 * h);
                        (void)Ep; // used indirectly via energy below

                        double gnorm = std::sqrt(grad_g * grad_g + grad_b * grad_b);
                        if (gnorm < tol) {
                            conv = true;
                            break;
                        }

                        adamStep(gk, bk, grad_g, grad_b, st, lr, beta1, beta2, eps);
                    }

                    finalParams[k * 2 + 0] = gk;
                    finalParams[k * 2 + 1] = bk;
                    converged[k] = conv ? 1 : 0;
                    epochsUsed[k] = static_cast<int32_t>(ep);
                    finalEnergy[k] = evaluateEnergy(graph, k, gk, bk);
                    if (conv)
                        ++convCount;
                }

                // ── Pack and annotate ─────────────────────────────────────────
                auto paramType =
                    RankedTensorType::get({static_cast<int64_t>(numSP), 2}, builder.getF64Type());
                auto paramAttr =
                    DenseElementsAttr::get(paramType, llvm::ArrayRef<double>(finalParams));

                auto energyType =
                    RankedTensorType::get({static_cast<int64_t>(numSP)}, builder.getF64Type());
                auto energyAttr =
                    DenseElementsAttr::get(energyType, llvm::ArrayRef<double>(finalEnergy));

                op->setAttr("warmstart_params", paramAttr);
                op->setAttr("warmstart_converged", DenseI32ArrayAttr::get(ctx, converged));
                op->setAttr("warmstart_epochs_used", DenseI32ArrayAttr::get(ctx, epochsUsed));
                op->setAttr("warmstart_final_energy", energyAttr);

                // Remark
                llvm::SmallString<200> info;
                llvm::raw_svector_ostream ss(info);
                ss << "doqaoa-warmstart-scheduler: " << warmCount << " warm-start sub-problems — "
                   << convCount << "/" << warmCount << " converged"
                   << " (tol=" << llvm::format("%.0e", tol) << " max_epochs=" << maxEp << ")";
                op->emitRemark() << info;
            });
    }
};

} // namespace quantum
} // namespace catalyst
