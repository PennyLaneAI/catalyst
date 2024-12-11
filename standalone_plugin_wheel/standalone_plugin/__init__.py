import platform
from pathlib import Path

StandalonePluginPathExt = "so" if platform.system() == "Linux" else "dylib"
StandalonePluginPath = Path(
    Path(__file__).parent, f"lib/StandalonePlugin.{StandalonePluginPathExt}"
)
SwitchBarToFoo = "standalone-switch-bar-foo"
