import platform
from pathlib import Path

import pennylane as qml

from catalyst.passes import PassPlugin


def SwitchBarToFoo(*flags, **valued_options):
    """Applies the "standalone-switch-bar-foo" pass"""

    SwitchBarToFoo.ext = "so" if platform.system() == "Linux" else "dylib"
    SwitchBarToFoo.path = Path(Path(__file__).parent, f"lib/StandalonePlugin.{SwitchBarToFoo.ext}")

    def decorator(qnode):
        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        def qnode_call(*args, **kwargs):
            pass_pipeline = kwargs.get("pass_pipeline", [])
            pass_pipeline.append(
                PassPlugin(
                    SwitchBarToFoo.path, "standalone-switch-bar-foo", *flags, **valued_options
                )
            )
            kwargs["pass_pipeline"] = pass_pipeline
            return qnode(*args, **kwargs)

        return qnode_call

    return decorator
