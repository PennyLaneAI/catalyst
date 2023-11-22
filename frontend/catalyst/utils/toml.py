import importlib.util

# TODO:
# Once Python version 3.11 is the oldest supported Python version, we can remove tomli
# and rely exclusively on tomllib.

# New in version 3.11
# https://docs.python.org/3/library/tomllib.html
tomllib = importlib.util.find_spec("tomllib")
tomlkit = importlib.util.find_spec("tomlkit")
# We need at least one of these to make sure we can read toml files.
if tomllib is None and tomlkit is None:
    msg = "Either tomllib or tomli need to be installed."
    raise ImportError(msg)

# Give preference to tomllib
if tomllib:
    from tomllib import load as toml_load
else:
    from tomlkit import load as toml_load

__all__ = ["toml_load"]
