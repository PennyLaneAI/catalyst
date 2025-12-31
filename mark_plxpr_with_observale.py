import pennylane as qml
import catalyst
from catalyst import qjit

import jax


qml.capture.enable()

@qml.qnode(qml.device("lightning.qubit", wires=2))
def circuit():
    qml.Hadamard(wires=0)
    return qml.expval(qml.X(0))


plxpr = jax.make_jaxpr(circuit)()
print(plxpr)

"""
{ lambda ; . let
    a:f64[] = qnode[
      device=<lightning.qubit device (wires=2) at 0x7c867738a750>
      execution_config=ExecutionConfig(grad_on_execution=False, use_device_gradient=None, use_device_jacobian_product=False, gradient_method='best', gradient_keyword_arguments={}, device_options={}, interface=<Interface.JAX: 'jax'>, derivative_order=1, mcm_config=MCMConfig(mcm_method=None, postselect_mode=None), convert_to_numpy=True, executor_backend=<class 'pennylane.concurrency.executors.native.multiproc.MPPoolExec'>)
      n_consts=0
      qfunc_jaxpr={ lambda ; . let
          _:AbstractOperator() = Hadamard[n_wires=1] 0:i64[]
          b:AbstractOperator() = PauliX[n_wires=1] 0:i64[]
          c:AbstractMeasurement(n_wires=None) = expval_obs b
        in (c,) }
      qnode=<QNode: device='<lightning.qubit device (wires=2) at 0x7c867738a750>', interface='jax', diff_method='best', shots='Shots(total=None)'>
      shots_len=0
    ]
  in (a,) }

Importantly, both gates and observables share the same primitive in plxpr: the "operator primitive".
https://github.com/PennyLaneAI/pennylane/blob/b301733ee59d6cf072e865bb5ad35dc8b5b2e2ae/pennylane/operation.py#L323

The difference is that the "observable" primitive kind returns a used value, whereas the "gate" primitive kind returns an unused value.

However, during jaxpr-to-mlir lowering, this information is not available, since that lowering is an interpreter.
i.e. During the lowering of PauliX eqn, the expval eqn does not exist yet.

So we need to mark the PauliX eqn with a kwarg to say that it should be lowered to an NamedObsOp
(instead of CustomOp).

So, before the lowering interpreter, there must be a preprocessing of plxpr to mark them.
"""


def var_is_AbstractOperator(var: jax._src.core.Var):
    return isinstance(var.aval, qml.operation._get_abstract_operator())

def mark_obs(jaxpr):

    var_to_defining_eqns = {}

    for eqn in jaxpr.eqns:

        # 1. Save all operator primitives. We do this to avoid a O(n^2) search
        # We need to search manually since unfortunately jaxpr values does not have a concept of "user"
        # ... or does it? I don't know. I can't find one.

        if getattr(eqn.primitive, "prim_type", "") == "operator":
            # Initialize param on all operator primitives
            eqn.params["is_observable"] = False

            if all(isinstance(v, jax._src.core.DropVar) for v in eqn.outvars):
                # "gate" primitives return dropvars in plxpr, won't be observables
                continue

            # Having non-dropvar outvars means it might be an observable
            # It could also be a inner eqn to an higher-order primitive like adjoint, hence "might"
            # But it is a candidate for being an observable
            for v in eqn.outvars:
                if var_is_AbstractOperator(v):
                    var_to_defining_eqns[v] = eqn


        # 2. Look for operator primitives used by measurement primitives
        # This "step 2" is guaranteed to occur after the above "step 1", even in the same loop!
        # This is because measurement primitives are guaranteed to occur after operator primitives.
        # But of course we can do two loops if people are paranoid.

        elif getattr(eqn.primitive, "prim_type", "") == "measurement":
            for v in eqn.invars:
                if var_is_AbstractOperator(v):
                    var_to_defining_eqns[v].params["is_observable"] = True



mark_obs(plxpr.eqns[0].params["qfunc_jaxpr"])
print(plxpr)

"""
{ lambda ; . let
    a:f64[] = qnode[
      device=<lightning.qubit device (wires=2) at 0x7846a2dfb980>
      execution_config=ExecutionConfig(grad_on_execution=False, use_device_gradient=None, use_device_jacobian_product=False, gradient_method='best', gradient_keyword_arguments={}, device_options={}, interface=<Interface.JAX: 'jax'>, derivative_order=1, mcm_config=MCMConfig(mcm_method=None, postselect_mode=None), convert_to_numpy=True, executor_backend=<class 'pennylane.concurrency.executors.native.multiproc.MPPoolExec'>)
      n_consts=0
      qfunc_jaxpr={ lambda ; . let
          _:AbstractOperator() = Hadamard[is_observable=False n_wires=1] 0:i64[]
          b:AbstractOperator() = PauliX[is_observable=True n_wires=1] 0:i64[]
          c:AbstractMeasurement(n_wires=None) = expval_obs b
        in (c,) }
      qnode=<QNode: device='<lightning.qubit device (wires=2) at 0x7846a2dfb980>', interface='jax', diff_method='best', shots='Shots(total=None)'>
      shots_len=0
    ]
  in (a,) }

TADA!
"""
