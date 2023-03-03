import sys
import vl_convert as vlc
import altair as alt
import pandas as pd
from os import system, makedirs
from os.path import isfile, dirname
from json import load as json_load
from argparse import ArgumentParser, BooleanOptionalAction, Namespace as ParsedArguments
from typing import Tuple, Iterable
from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager
from hashlib import sha256

from pandas import DataFrame
from altair import Chart

from catalyst_benchmark.types import Sysinfo, BenchmarkResult
from catalyst_benchmark.main import parse_implementation

# fmt:off
FMTVERSION = 1
""" Version of serialized representation."""

SYSINFO = Sysinfo.fromOS()
SYSHASH = sha256(str(SYSINFO).encode("utf-8")).hexdigest()[:6]

CATPROBLEMS = {
    "regular": "grover",
    "deep": "grover",
    "hybrid": None,
    # "variational": "vqe",
    "variational": "chemvqe",
}

MINQUBITS = 7
MAXQUBITS = 29
QUBITS = {
    ("regular", "grover", "compile"): [MINQUBITS, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, MAXQUBITS],
    ("regular", "grover", "runtime"): [MINQUBITS, 9, 11, 13, 15, 17],
    ("deep", "grover", "compile"): [7],
    ("deep", "grover", "runtime"): [7],
    ("variational", "vqe", "compile"): [6, 7, 8, 9, 10, 11],
    ("variational", "vqe", "runtime"): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ("variational", "chemvqe", "runtime"): [4, 6, 8, 12],
}

MAXLAYERS = 1500
LAYERS = {
    ("deep", "grover", "compile"):
        [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, MAXLAYERS],
    ("deep", "grover", "runtime"):
        [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, MAXLAYERS],
}

KNOWN_FAILURES = {
    ("grover", "compile", "pennylane+jax/default.qubit.jax", None): (7, 50),
    ("grover", "runtime", "pennylane+jax/default.qubit.jax", None): (7, 50),
}

DIFF_METHODS = {
    "grover": [None],
    "vqe": ["finite-diff", "parameter-shift", "adjoint", "backprop"],
    "chemvqe": ["finite-diff", "parameter-shift", "adjoint", "backprop"]
}

# Implementation aliases to workaround the Altair clipped legend problem
# https://github.com/vega/vl-convert/issues/30
ALIASES = {
    "catalyst/lightning.qubit": "C/L",
    "pennylane+jax/lightning.qubit": "PLjax/L",
    "pennylane+jax/default.qubit.jax": "PLjax/Def",
    "pennylane/default.qubit": "PL/Def",
    "pennylane/lightning.qubit": "PL/L",
}

# Colors obtained from a Vega colorscheme.
# Ref. https://stackoverflow.com/questions/70993559/altair-selecting-a-color-from-a-vega-color-scheme-for-plot
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
          "#ffff33", "#a65628", "#f781bf", "#999999"]
# fmt:on


def ofile(a, _, measure, problem, impl, nqubits, nlayers, diffmethod) -> Tuple[str, str]:
    """Produce the JSON file name containing the measurement configured and
    the Linux shell command which is expected to produce such file."""
    impl_ = impl.replace("+", "_").replace("/", "_").replace(".", "_")
    dmfilepart = f"_{diffmethod}".replace("-", "_") if diffmethod is not None else ""
    ofname = (
        f"_benchmark/{measure}_{problem}_{impl_}{dmfilepart}_N{nqubits}_"
        f"L{nlayers}_S{SYSHASH}_v{FMTVERSION}/results.json"
    )
    if problem == "grover":
        assert diffmethod is None
        params = f"--grover-nlayers={nlayers}" if nlayers is not None else ""
    elif problem == "vqe" or problem == "chemvqe":
        assert nlayers is None
        assert diffmethod is not None
        params = f"--vqe-diff-method={diffmethod}"
    else:
        raise ValueError(f"Unsupported problem {problem}")
    cmdline = (
        f"python3 -m catalyst_benchmark.main run "
        f"--problem={problem} "
        f"--measure={measure} "
        f"--implementation={impl} "
        f"--nqubits={nqubits} "
        f"--output={ofname} "
        f"--timeout={a.timeout_1run} "
        f"{params}"
    )
    return (ofname, cmdline)


def loadresults(fp: str) -> BenchmarkResult:
    """Load a serrialized benchmark result from file"""
    return BenchmarkResult.from_dict(json_load(open(fp)))


def all_configurations(a: ParsedArguments) -> Iterable[tuple]:
    """Iterate through the configurations available."""

    for measure in ["compile", "runtime"]:
        for impl in [
            "catalyst/lightning.qubit",
            "pennylane+jax/default.qubit.jax",
            "pennylane+jax/lightning.qubit",
            "pennylane/default.qubit",
            "pennylane/lightning.qubit",
        ]:
            framework, _, _ = parse_implementation(impl)

            for cat in ["regular", "deep", "hybrid", "variational"]:
                if not any([(c in a.measure) for c in [measure, "all"]]):
                    continue
                if not any([(c in a.category) for c in [cat, "all"]]):
                    continue

                problem = CATPROBLEMS.get(cat, None)
                if problem is None:
                    continue

                for diffmethod in DIFF_METHODS[problem]:
                    for nqubits in sorted(QUBITS.get((cat, problem, measure), [None])):
                        for nlayers in sorted(LAYERS.get((cat, problem, measure), [None])):
                            yield (cat, measure, problem, impl, nqubits, nlayers, diffmethod)


def collect(a: ParsedArguments) -> None:
    """Run the selected configurations and check for results. Avoid trying
    larger configurations if smaller configurations failed. In the end, print
    the `known_failures` dictionary suggestion."""
    known_failures = deepcopy(KNOWN_FAILURES)
    try:
        for config in all_configurations(a):
            ofname, cmdline = ofile(a, *config)
            odname = dirname(ofname)

            (cat, measure, problem, impl, nqubits, nlayers, diffmethod) = config
            if len(odname) > 0 and not a.dry_run:
                makedirs(odname, exist_ok=True)
            cmdline += f" 2>&1 | tee {odname}/output.log"
            if isfile(ofname):
                print(f"Skipping existing: {cmdline}")
            else:
                if a.dry_run:
                    print(f"(Dry-run) Would run: {cmdline}")
                else:
                    (fnqubits, fnlayers) = known_failures.get(
                        (problem, measure, impl, diffmethod), (sys.maxsize, sys.maxsize)
                    )
                    if (nqubits, nlayers) >= (fnqubits, fnlayers):
                        print(f"Skipping as likely to fail: {cmdline}")
                    else:
                        print(f"Running: {cmdline}")
                        system(cmdline)
                        if not isfile(ofname):
                            known_failures[(problem, measure, impl, diffmethod)] = (
                                nqubits,
                                nlayers,
                            )
    finally:
        if str(known_failures) != str(KNOWN_FAILURES):
            print("Suggestion:\nKNOWN_FAILURES =")
            print(known_failures)


def load(a: ParsedArguments) -> DataFrame:
    """Load the benchmark data into the Pandas DataFrame"""
    log = []
    nmissing = 0
    data = defaultdict(list)
    for config in all_configurations(a):
        ofname, _ = ofile(a, *config)
        cat, measure, problem, impl, nqubits, nlayers, diffmethod = config
        r = None
        try:
            r = loadresults(ofname)
        except Exception as e:
            nmissing += 1
            log.append(str(e))
        if r is not None:
            for trial, time in enumerate(r.measurement_sec):
                data["cat"].append(cat)
                data["measure"].append(measure)
                data["problem"].append(problem)
                data["trial"].append(trial)
                data["impl"].append(ALIASES[impl])
                data["nqubits"].append(nqubits)
                data["time"].append(time)
                data["nlayers"].append(nlayers)
                data["ngates"].append(r.depth_gates)
                data["diffmethod"].append(diffmethod)
    if nmissing > 0:
        print(f"There are {nmissing} data records missing", file=sys.stderr)
        if a.verbose:
            print("\n".join(log), file=sys.stderr)
        else:
            print("Pass -V to see the full list", file=sys.stderr)
    return DataFrame(data)


def plot(a: ParsedArguments) -> None:
    """Plot the figures. The function first builds a set of Pandas DataFrames,
    then calls Altair to present the data collected."""
    df_full = load(a)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        implCLcond = alt.condition("datum.impl == 'C/L'", alt.value(2), alt.value(0.7))
        implCLcondDash = alt.condition("datum.impl == 'C/L'", alt.value([0]), alt.value([3, 3]))

        @contextmanager
        def _open(fname: str, fmode: str):
            if a.dry_run and "w" in fmode:

                class DummyFile:
                    def write(*args, **kwargs):
                        return

                yield DummyFile()
                print(f"(Dry-run) Would update: {fname}")
            else:
                with open(fname, fmode) as f:
                    yield f

        def _implEncoding(df):
            """Calculate domain and range colors of the implementation, based on
            the actual dataset and the pre-defined palette."""
            dom, rang = zip(
                *[
                    (i, v)
                    for (i, v) in zip(ALIASES.values(), COLORS)
                    if i in set(df["impl"].to_list())
                ]
            )
            return alt.Color(
                "impl:N",
                title="Impl",
                legend=alt.Legend(columns=1, labelLimit=240),
                scale=alt.Scale(domain=dom, range=rang),
            )

        trialEncoding = alt.Opacity(
            "trial", title="Trial", legend=alt.Legend(columns=1, labelLimit=240)
        )

        timeEncoding = alt.Y("time", title="Time, sec", scale=alt.Scale(type="log"))

        def _nqubitsEncoding(title="# Qubits", **kwargs):
            return alt.X("nqubits", title=title, **kwargs)

        def _nlayersEncoding(title="# Layers", **kwargs):
            return alt.X("nlayers", title=title, **kwargs)

        def _mktitle(s: str) -> dict:
            return {
                "text": s,
                "subtitle": SYSINFO.toString(),
                "subtitleFontSize": 9,
                "align": "center",
            }

        def _mkfooter(df: DataFrame, xEncoding, ref_impl: str) -> Chart:
            df_gate = df[(df["trial"] == 0) & (df["impl"] == ref_impl)]
            return (
                Chart(df_gate)
                .mark_area()
                .encode(
                    x=xEncoding,
                    y=alt.Y("ngates", title="# Gates", axis=alt.Axis(grid=False)),
                    color=alt.value("grey"),
                    opacity=alt.value(0.4),
                )
                .properties(height=60)
            )

        def _filter(cat, measure, problem):
            return df_full[(df_full["cat"]==cat) &
                           (df_full["measure"]==measure) &
                           (df_full["problem"]==problem)]

        df = _filter("regular", "compile", "grover")
        if len(df) > 0:
            xaxis = alt.Axis(values=list(range(MINQUBITS, MAXQUBITS + 2, 2)))
            xscale = alt.Scale(domain=(MINQUBITS, MAXQUBITS))
            print("Updating _img/regular_compile.svg")
            with _open("_img/regular_compile.svg", "w") as f:
                f.write(
                    vlc.vegalite_to_svg(
                        alt.vconcat(
                            Chart(df)
                            .mark_line(point=True)
                            .encode(
                                x=_nqubitsEncoding(title=None, axis=xaxis, scale=xscale),
                                y=timeEncoding,
                                color=_implEncoding(df),
                                opacity=trialEncoding,
                                strokeDash=implCLcondDash,
                                strokeWidth=implCLcond,
                            )
                            .properties(title=_mktitle("Compilation time, Regular circuits")),
                            _mkfooter(df, _nqubitsEncoding(axis=xaxis, scale=xscale), "PL/L"),
                        )
                        .configure_axisLeft(minExtent=50)
                        .to_dict()
                    ),
                )

        df = _filter("regular", "runtime", "grover")
        if len(df) > 0:
            print("Updating _img/regular_runtime.svg")
            with _open("_img/regular_runtime.svg", "w") as f:
                f.write(
                    vlc.vegalite_to_svg(
                        alt.vconcat(
                            Chart(df)
                            .mark_line(point=True)
                            .encode(
                                x=_nqubitsEncoding(title=None),
                                y=timeEncoding,
                                opacity=trialEncoding,
                                color=_implEncoding(df),
                                strokeDash=implCLcondDash,
                                strokeWidth=implCLcond,
                            )
                            .properties(title=_mktitle("Running time, Regular circuits")),
                            _mkfooter(df, _nqubitsEncoding(), "PL/L"),
                        )
                        .configure_axisLeft(minExtent=50)
                        .to_dict()
                    ),
                )

        df = _filter("deep", "compile", "grover")
        if len(df) > 0:
            xaxis = alt.Axis(values=list(range(0, MAXLAYERS + 100, 100)))
            xscale = alt.Scale(domain=(0, MAXLAYERS))
            print("Updating _img/deep_compile.svg")
            with _open("_img/deep_compile.svg", "w") as f:
                f.write(
                    vlc.vegalite_to_svg(
                        alt.vconcat(
                            Chart(df)
                            .mark_line(point=True)
                            .encode(
                                x=_nlayersEncoding(title=None, axis=xaxis, scale=xscale),
                                y=timeEncoding,
                                color=_implEncoding(df),
                                opacity=trialEncoding,
                                strokeDash=implCLcondDash,
                                strokeWidth=implCLcond,
                            )
                            .properties(title=_mktitle("Compilation time, Deep circuits")),
                            _mkfooter(df, _nlayersEncoding(axis=xaxis, scale=xscale), "PLjax/L"),
                        )
                        .configure_axisLeft(minExtent=50)
                        .to_dict()
                    ),
                )

        df = _filter("deep", "runtime", "grover")
        if len(df) > 0:
            xaxis = alt.Axis(values=list(range(0, MAXLAYERS + 100, 100)))
            xscale = alt.Scale(domain=(0, MAXLAYERS))
            print("Updating _img/deep_runtime.svg")
            with _open("_img/deep_runtime.svg", "w") as f:
                f.write(
                    vlc.vegalite_to_svg(
                        alt.vconcat(
                            Chart(df)
                            .mark_line(point=True)
                            .encode(
                                x=_nlayersEncoding(title=None, axis=xaxis, scale=xscale),
                                y=timeEncoding,
                                color=_implEncoding(df),
                                opacity=trialEncoding,
                                strokeDash=implCLcondDash,
                                strokeWidth=implCLcond,
                            )
                            .properties(title=_mktitle("Running time, Deep circuits")),
                            _mkfooter(df, _nlayersEncoding(axis=xaxis, scale=xscale), "PL/L"),
                        )
                        .configure_axisLeft(minExtent=50)
                        .to_dict()
                    ),
                )

        df = _filter("variational", "compile", "vqe")
        if len(df) > 0:
            print("Updating _img/variational_compile.svg")
            with _open("_img/variational_compile.svg", "w") as f:
                f.write(
                    vlc.vegalite_to_svg(
                        Chart(df)
                        .mark_line(point=True)
                        .encode(
                            x=_nqubitsEncoding(),
                            y=timeEncoding,
                            color=_implEncoding(df),
                            opacity=trialEncoding,
                            strokeDash=implCLcondDash,
                            strokeWidth=implCLcond,
                        )
                        .properties(title=_mktitle("Compilaiton time, Variational circuits"))
                        .to_dict()
                    ),
                )

        problem = "chemvqe"
        df_allgrad = _filter("variational", "runtime", problem)
        for diffmethod in DIFF_METHODS[problem]:
            df = df_allgrad[df_allgrad["diffmethod"]==diffmethod]
            if len(df) > 0:
                fname = f"_img/variational_runtime_{diffmethod.replace('-','')}.svg"
                print(f"Updating {fname}")
                with _open(fname, "w") as f:
                    f.write(
                        vlc.vegalite_to_svg(
                            # alt.vconcat(
                            Chart(df)
                            # .transform_calculate(impl_diff="datum.impl+'('+datum.diffmethod+')'")
                            .mark_line(point=True)
                            .encode(
                                x=_nqubitsEncoding(),
                                y=timeEncoding,
                                # color=_implEncoding(df),
                                color=_implEncoding(df),
                                opacity=trialEncoding,
                                strokeDash=implCLcondDash,
                                strokeWidth=implCLcond,
                            )
                            .properties(title=_mktitle(
                                f"Running time, Variational circuits ({diffmethod})"))
                            .to_dict(),
                            # _mkfooter(df, _nqubitsEncoding(), "PL/Def")
                            # )
                            # .configure_axisLeft(minExtent=50)
                            # .to_dict(),
                        ),
                    )


# fmt:off
ap = ArgumentParser(prog="batchrun.py")
ap.add_argument("-m", "--measure", type=str, default="all",
                help="Value to measure: compile|runtime|all, (default - 'all')")
ap.add_argument("-c", "--category", type=str, default="regular,deep",
                help=("Category of circutis to evaluate regular|deep|hybrid|variational|all "
                "(default - 'regular,deep')"))
ap.add_argument("--dry-run", default=False, action=BooleanOptionalAction,
                help="Enable this mode to print command lines but not actually run anything")
ap.add_argument("-a", "--actions", type=str, default="collect,plot",
                help="Which actions to perform (default - 'collect,plot')")
ap.add_argument("-t", "--timeout-1run", type=str, metavar="SEC", default="1000.0",
                help="Timeout for single benchmark run (default - 1000)")
ap.add_argument("-V", "--verbose", default=False, action=BooleanOptionalAction,
                help="Print verbose messages")
# fmt: on

def load_all() -> DataFrame:
    return load(ap.parse_args([]))

if __name__ == "__main__":
    a = ap.parse_args(sys.argv[1:])

    if "collect" in a.actions:
        collect(a)
    else:
        print("Skipping the 'collect' action")
    if "plot" in a.actions:
        plot(a)
    else:
        print("Skipping teh 'plot' action")
