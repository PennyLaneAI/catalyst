# Catalyst Python build control plane

Cross-platform (Linux / macOS / **Windows**) replacement for the recursive GNU
Makefiles that drive the Catalyst build. It calls CMake / Ninja / pip directly
from Python with **no shell dependency** (no `bash`, `sh`, `make`, `uname`,
`cp`, `rm`, `find`, `mkdir`, `echo`), so the same commands work on Windows
without WSL / MSYS / Git-Bash.

## Usage

Run from the repo root, exactly where you used to run `make`:

```
python build.py <target> [<target> ...] [options]
```

The designated validation/test point is `frontend` (identical to `make frontend`):

```
python build.py frontend
```

Preview any target without executing (prints every command and filesystem op):

```
python build.py --dry-run all
```

### Common examples

```
python build.py all                       # runtime oqc mlir frontend
python build.py catalyst                  # runtime dialects plugin frontend oqc
python build.py runtime --enable-oqd
python build.py llvm --build-type Release -j 8
python build.py wheel
python build.py clean-all
```

`python build.py --help` lists every target and flag.

## How configuration is resolved

Each knob can be set three ways, in increasing priority:

1. Built-in default (matches the Makefile default exactly).
2. Environment variable (the Makefile `?=` overrides still work, e.g.
   `CMAKE=/path/to/cmake`, `ENABLE_OQD=ON`, `BUILD_TYPE=Release`, `NPROC=8`).
3. A command-line flag (e.g. `--cmake`, `--enable-oqd`, `--build-type`, `-j`).

Toolchain discovery uses `shutil.which` (resolves `.exe`/`.bat`/`.cmd` via
`PATHEXT` on Windows). Compiler preference is clang → cl → gcc; LLD is enabled
by default everywhere except macOS, matching `mlir/Makefile`.

## Layout

```
build.py                      # root launcher (python build.py <target>)
catalyst_build/
  cli.py                      # argparse dispatcher: target name -> function
  environment.py              # BuildEnv: tool discovery + all config knobs + paths
  runner.py                   # portable run()/cp/rm/find/mkdir helpers (no shell)
  wheel.py                    # `wheel` and `plugin-wheel` targets
  components/
    frontend.py               # `frontend` (test point) + `clean`
    runtime.py                # runtime configure/build/test/clean
    mlir.py                   # llvm/stablehlo/enzyme/dialects/docs/plugin + cleans
    oqc.py                    # OQC device build/test/clean
```

## Target mapping (Makefile -> build.py)

| Makefile                  | build.py                      |
|---------------------------|-------------------------------|
| `make all`                | `python build.py all`         |
| `make catalyst`           | `python build.py catalyst`    |
| `make frontend`           | `python build.py frontend`    |
| `make mlir`               | `python build.py mlir`        |
| `make llvm`               | `python build.py llvm`        |
| `make stablehlo`          | `python build.py stablehlo`   |
| `make enzyme`             | `python build.py enzyme`      |
| `make dialects`           | `python build.py dialects`    |
| `make dialect-docs`       | `python build.py dialect-docs`|
| `make plugin`             | `python build.py plugin`      |
| `make runtime`            | `python build.py runtime`     |
| `make oqc`                | `python build.py oqc`         |
| `make builtin-decomp-rules` | `python build.py builtin-decomp-rules` |
| `make wheel`              | `python build.py wheel`       |
| `make plugin-wheel`       | `python build.py plugin-wheel`|
| `make test-runtime`       | `python build.py test-runtime`|
| `make test-mlir`          | `python build.py test-mlir`   |
| `make test-oqc`           | `python build.py test-oqc`    |
| `make clean`              | `python build.py clean`       |
| `make clean-all`          | `python build.py clean-all`   |
| `make clean-mlir` / `-dialects` / `-llvm` / `-stablehlo` / `-enzyme` / `-plugin` / `-runtime` / `-oqc` | same names |
| `make reset-llvm`         | `python build.py reset-llvm`  |

## Notes / not-yet-ported

These Makefile targets are POSIX-shell- or external-tool-specific and were left
out of the first cut (they are not part of the build itself):

- `lit` / `pytest` / `test-frontend` / `test-demos` (Python test orchestration,
  including the `CATALYST_LIBPYTHON` lit plumbing and ASAN-with-clang guard).
- `coverage*` (lcov / gcov post-processing).
- `format` / `format-frontend` (clang-format / black / isort wrappers).
- `docs` / `test-toml-spec` (Sphinx + `bin/toml-check.py`).

The hooks for these (`env.libpython_path()`, `env.using_clang()`,
`bin/utils.py`) already exist, so they are straightforward follow-ons.
