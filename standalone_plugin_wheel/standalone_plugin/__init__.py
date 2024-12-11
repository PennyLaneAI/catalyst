import platform
from pathlib import Path

from catalyst.passes import PassPlugin


def SwitchBarToFoo(*flags, **valued_options):
    """Applies the "standalone-switch-bar-foo" pass"""

    SwitchBarToFoo.ext = "so" if platform.system() == "Linux" else "dylib"
    SwitchBarToFoo.path = Path(Path(__file__).parent, f"lib/StandalonePlugin.{SwitchBarToFoo.ext}")

    def decorator(qnode):

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
