import platform
from functools import partial
from pathlib import Path

from catalyst.passes import apply_pass_plugin

SwitchBarToFooExt = "so" if platform.system() == "Linux" else "dylib"
SwitchBarToFooPath = Path(Path(__file__).parent, f"lib/StandalonePlugin.{SwitchBarToFooExt}")
SwitchBarToFoo = partial(apply_pass_plugin, SwitchBarToFooPath, "standalone-switch-bar-foo")
