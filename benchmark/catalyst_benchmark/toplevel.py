""" Measurement cycle management and data post-processing routines """
import sys
from os import system, makedirs
from os.path import isfile, dirname
from json import load as json_load
from argparse import ArgumentParser, Namespace as ParsedArguments
from typing import Tuple, Iterable, Set, Union, Optional
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from itertools import chain
from functools import partial

import vl_convert as vlc
import altair as alt
import pandas as pd
from pandas import DataFrame
from altair import Chart

from catalyst_benchmark.types import (
    Sysinfo,
    BenchmarkResult,
    BenchmarkResultV1,
    BooleanOptionalAction,
)
from catalyst_benchmark.measurements import parse_implementation


# fmt:off
FMTVERSION = 1
""" Version of serialized representation."""

SYSINFO = Sysinfo.fromOS()
SYSHASH_ORIG = sha256(str(SYSINFO).encode("utf-8")).hexdigest()[:6]
SYSHASH = SYSHASH_ORIG

IMPLEMENTATIONS = [
    "catalyst/lightning.qubit",
    "pennylane+jax/default.qubit.jax",
    "pennylane+jax/lightning.qubit",
    "pennylane/default.qubit",
    "pennylane/lightning.qubit",
]

# Implementation aliases to workaround the Altair clipped legend problem
# https://github.com/vega/vl-convert/issues/30
ALIASES = {
    "catalyst/lightning.qubit": "C/L",
    "pennylane+jax/lightning.qubit": "PLjax/L",
    "pennylane/lightning.qubit": "PL/L",
    "pennylane+jax/default.qubit.jax": "PLjax/Def",
    "pennylane/default.qubit": "PL/Def",
}

C_L = ALIASES["catalyst/lightning.qubit"]
PL_L = ALIASES["pennylane/lightning.qubit"]
PLjax_L = ALIASES["pennylane+jax/lightning.qubit"]

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
    ("deep", "qft", "compile"): [7],
    ("deep", "qft", "runtime"): [7],
    ("variational", "vqe", "compile"): [6, 7, 8, 9, 10, 11],
    ("variational", "vqe", "runtime"): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ("variational", "chemvqe", "compile"): [4, 6, 8, 12],
    ("variational", "chemvqe", "runtime"): [4, 6, 8, 12],
}

MAXLAYERS = 1500
LAYERS = {
    ("deep", "grover", "compile"):
        [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, MAXLAYERS],
    ("deep", "grover", "runtime"):
        [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, MAXLAYERS],
    ("deep", "qft", "compile"):
        [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, MAXLAYERS],
    ("deep", "qft", "runtime"):
        [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, MAXLAYERS],
}

KNOWN_FAILURES = {
    # ("grover", "compile", "pennylane+jax/default.qubit.jax", None): (7, 50),
    # ("grover", "runtime", "pennylane+jax/default.qubit.jax", None): (7, 50),
}

DIFF_METHODS = {
    ("chemvqe","compile"): ["finite-diff", "parameter-shift", "adjoint", "backprop"],
    ("chemvqe","runtime"): ["finite-diff", "parameter-shift", "adjoint", "backprop"]
}

# Colors obtained from a Vega colorscheme.
# Ref. https://stackoverflow.com/questions/70993559/altair-selecting-a-color-from-a-vega-color-scheme-for-plot
COLORS = ["#e41a1c", "#377eb8", "#ff7f00", "#4daf4a", "#984ea3",
          "#ffff33", "#a65628", "#f781bf", "#999999"]

MEASUREMENTS = ["compile", "runtime", "compile-circuit"]

# fmt:on


def tag(a) -> str:
    """Format the user-defined part of data file names"""
    return a.tag if a.tag is not None else f"v{FMTVERSION}"


def syshash(a) -> str:
    """Format the system-information part of the data file names"""
    return a.force_sysinfo_hash if a.force_sysinfo_hash else SYSHASH


def ofile(a, _, measure, problem, impl, nqubits, nlayers, diffmethod) -> Tuple[str, str]:
    """Produce the JSON file name containing the measurement configured and
    the Linux shell command which is expected to produce such file."""
    measure_ = measure.replace("-","")
    impl_ = impl.replace("+", "_").replace("/", "_").replace(".", "_")
    dmfilepart = f"_{diffmethod}".replace("-", "_") if diffmethod is not None else ""
    ofname = (
        f"_benchmark/{measure_}_{problem}_{impl_}{dmfilepart}_N{nqubits}_"
        f"L{nlayers}_S{syshash(a)}_{tag(a)}/results.json"
    )
    if problem == "grover":
        assert diffmethod is None
        params = f"--nlayers={nlayers}" if nlayers is not None else ""
    elif problem == "qft":
        assert diffmethod is None
        params = f"--nlayers={nlayers}" if nlayers is not None else ""
    elif problem == "vqe" or problem == "chemvqe":
        assert nlayers is None
        assert diffmethod is not None
        params = f"--vqe-diff-method={diffmethod}"
    else:
        raise ValueError(f"Unsupported problem {problem}")
    cmdline = (
        f"python3 benchmark.py run "
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
    return BenchmarkResult.from_dict(json_load(open(fp, encoding="utf-8")))


def all_configurations(a: ParsedArguments) -> Iterable[tuple]:
    """Iterate through the configurations available."""

    for measure in MEASUREMENTS:
        for impl in IMPLEMENTATIONS:
            framework, _, _ = parse_implementation(impl)

            for cat in ["regular", "deep", "hybrid", "variational"]:
                if not any([(m in a.measure.split(',')) for m in [measure, "all"]]):
                    continue
                if not any([(c in a.category.split(',')) for c in [cat, "all"]]):
                    continue

                problem = CATPROBLEMS.get(cat, None)
                if problem is None:
                    continue

                for diffmethod in DIFF_METHODS.get((problem, measure),[None]):
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
            pdesc = f"{problem}[{nqubits},{nlayers}]"
            hint = f"{measure: <15} {impl: <32} {cat: <15} {pdesc: <20}"
            message = hint if not a.verbose else cmdline
            if isfile(ofname):
                print(f"{message} [EXISTS]")
            else:
                if a.dry_run:
                    print(f"{message} [DRYRUN]")
                else:
                    (fnqubits, fnlayers) = known_failures.get(
                        (problem, measure, impl, diffmethod), (None, None)
                    )
                    if (nqubits or 0, nlayers or 0) >= (
                        fnqubits or sys.maxsize,
                        fnlayers or sys.maxsize,
                    ):
                        print(f"{message} [WOULDFAIL]")
                    else:
                        print(f"{message}", end='', flush=True)
                        system(cmdline)
                        if not isfile(ofname):
                            known_failures[(problem, measure, impl, diffmethod)] = (
                                nqubits,
                                nlayers,
                            )
                            print(" [FAIL]")
                        else:
                            print(" [OK]")
    finally:
        if str(known_failures) != str(KNOWN_FAILURES):
            print("Suggestion:\nKNOWN_FAILURES =")
            print(known_failures)


def load(a: ParsedArguments) -> Tuple[DataFrame, Optional[Sysinfo]]:
    """Load the benchmark data into the Pandas DataFrame"""
    log = []
    nmissing = 0
    systems = set()
    data = defaultdict(list)
    for config in all_configurations(a):
        ofname, _ = ofile(a, *config)
        cat, measure, problem, impl, nqubits, nlayers, diffmethod = config
        r = None
        try:
            r = BenchmarkResult.from_dict(json_load(open(ofname, encoding="utf-8")))
        except Exception as e:
            log.append(str(e))
            log.append("Trying to load V1 instead")
            try:
                r = BenchmarkResultV1.from_dict(json_load(open(ofname, encoding="utf-8")))
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
                timeout_ntrials = (
                    r.timeout_sec
                    if hasattr(r, "timeout_sec")
                    else (float(a.timeout_1run) if a.timeout_1run else 1e9)
                )
                data["timeout"].append(timeout_ntrials / len(r.measurement_sec))
                systems.add(r.sysinfo)
    if nmissing > 0:
        print(f"There are {nmissing} data records missing", file=sys.stderr)
        if a.verbose:
            print("\n".join(log), file=sys.stderr)
        else:
            print("Pass -V to see the full list", file=sys.stderr)
    if len(systems) > 1:
        systems_str = '\n'.join([str(s) for s in systems])
        print(f"Data was collected from more than one system:\n{systems_str}", file=sys.stderr)
    return DataFrame(data), (list(systems)[0] if len(systems) > 0 else None)


def writefile(a: ParsedArguments, fname, chart) -> None:
    """Write chart to file(s) in the configured formats"""
    fname_suffix = f"{fname}_{syshash(a)}_{tag(a)}"
    for ext, wf, method in [("svg", "w", vlc.vegalite_to_svg), ("png", "wb", vlc.vegalite_to_png)]:
        if ext in a.plot_formats:
            if a.dry_run and "w" in wf:
                print(f"(Dry-run) Would update: {fname_suffix}.{ext}")
            else:
                print(f"Updating {fname_suffix}.{ext}")
                with open(f"{fname_suffix}.{ext}", wf, encoding="utf-8") as f:
                    f.write(method(chart))


def dfilter(cat, measure, problem, df: DataFrame) -> DataFrame:
    """Filter the benchmark data"""
    try:
        return df[(df["cat"] == cat) & (df["measure"] == measure) & (df["problem"] == problem)]
    except KeyError:
        return DataFrame()


def plot(a: ParsedArguments, df_full: DataFrame, sysinfo: Optional[Sysinfo] = None) -> None:
    """Plot the figures. The function first builds a set of Pandas DataFrames,
    then calls Altair to present the data collected."""

    _filter = partial(dfilter, df=df_full)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        implCLcond = alt.condition(f"datum.impl == '{C_L}'", alt.value(2), alt.value(0.7))
        implCLcondDash = alt.condition(f"datum.impl == '{C_L}'", alt.value([0]), alt.value([3, 3]))

        def _implEncoding(df, add_timeout=False, **kwargs):
            """Calculate domain and range colors of the implementation, based on
            the actual dataset and the pre-defined palette."""
            dom, rang = zip(
                *[
                    (i, v)
                    for (i, v) in zip(ALIASES.values(), COLORS)
                    if i in set(df["impl"].to_list())
                ]
            )
            if add_timeout:
                dom = list(dom) + ["Timeout"]
                rang = list(rang) + ["grey"]
            return alt.Color(
                "impl:N",
                title="Impl",
                legend=alt.Legend(columns=1, labelLimit=240),
                scale=alt.Scale(domain=dom, range=rang),
                **kwargs,
            )

        trialEncoding = alt.Opacity(
            "trial", title="Trial", legend=alt.Legend(columns=1, labelLimit=240)
        )

        def _timeEncoding(mean: bool = False):
            return alt.Y(
                "mean(time):Q" if mean else "time:Q",
                title="Mean time, sec" if mean else "Time, sec",
                scale=alt.Scale(type="log"),
            )

        def _nqubitsEncoding(title="Qubits", **kwargs):
            return alt.X("nqubits", title=title, **kwargs)

        def _nlayersEncoding(title="Layers", **kwargs):
            return alt.X("nlayers", title=title, **kwargs)

        def _mktitle(s: str, align="center") -> dict:
            t = {
                "text": s,
                "subtitle": sysinfo.toString() if sysinfo else "Unknown system",
                "subtitleFontSize": 9,
            }
            if align:
                t.update({"align": align})
            return t

        def _mkfooter(df: DataFrame, xEncoding, ref_impl: str) -> Chart:
            df_gate = df[(df["trial"] == 0) & (df["impl"] == ref_impl)]
            return (
                Chart(df_gate)
                .mark_area()
                .encode(
                    x=xEncoding,
                    y=alt.Y("ngates", title="Gates", axis=alt.Axis(grid=False)),
                    color=alt.value("grey"),
                    opacity=alt.value(0.4),
                )
                .properties(height=60)
            )

        def _plot_trial_linechart(df, fname, xenc, title, **kwargs):
            if len(df) == 0:
                return
            writefile(
                a,
                fname,
                Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("trial:N", title="Trials"),
                    y=_timeEncoding(),
                    color=_implEncoding(df),
                    opacity=xenc(**kwargs),
                )
                .properties(title=title)
                .to_dict(),
            )

        def _plot_linechart(df, fname, xenc, title, ref_ngates=None, **kwargs):
            if len(df) == 0:
                return
            writefile(
                a,
                fname,
                alt.vconcat(
                    *[
                        Chart(df)
                        .mark_line(point=True)
                        .encode(
                            x=xenc(title=None, **kwargs),
                            y=_timeEncoding(),
                            color=_implEncoding(df),
                            opacity=trialEncoding,
                            strokeDash=implCLcondDash,
                            strokeWidth=implCLcond,
                        )
                        .properties(title=title),
                    ]
                    + (
                        [
                            _mkfooter(df, xenc(**kwargs), ref_ngates),
                        ]
                        if ref_ngates is not None
                        else []
                    )
                )
                .configure_axisLeft(minExtent=50)
                .to_dict(),
            )

        # Regular circuits
        df = _filter("regular", "compile", CATPROBLEMS["regular"])
        xaxis = alt.Axis(values=list(range(MINQUBITS, MAXQUBITS + 2, 2)))
        xscale = alt.Scale(domain=(MINQUBITS, MAXQUBITS))
        _plot_linechart(
            df,
            "_img/regular_compile",
            _nqubitsEncoding,
            _mktitle("Compilation time, Regular circuits"),
            PL_L,
            axis=xaxis,
            scale=xscale,
        )
        _plot_trial_linechart(
            df,
            "_img/regular_compile_trial",
            _nqubitsEncoding,
            _mktitle("Compilation time/trial, Regular circuits"),
        )

        df = _filter("regular", "runtime", CATPROBLEMS["regular"])
        _plot_linechart(
            df,
            "_img/regular_runtime",
            _nqubitsEncoding,
            _mktitle("Running time, Regular circuits"),
            PL_L,
        )
        _plot_trial_linechart(
            df,
            "_img/regular_runtime_trial",
            _nqubitsEncoding,
            _mktitle("Running time/trial, Regular circuits"),
        )

        # Deep circuits
        df = _filter("deep", "compile", CATPROBLEMS["deep"])
        xaxis = alt.Axis(values=list(range(0, MAXLAYERS + 100, 100)))
        xscale = alt.Scale(domain=(0, MAXLAYERS))
        _plot_linechart(
            df,
            "_img/deep_compile",
            _nlayersEncoding,
            _mktitle("Compilation time, Deep circuits"),
            PLjax_L,
            axis=xaxis,
            scale=xscale,
        )
        _plot_trial_linechart(
            df,
            "_img/deep_compile_trial",
            _nlayersEncoding,
            _mktitle("Compilation time/trial, Deep circuits"),
        )

        df = _filter("deep", "runtime", CATPROBLEMS["deep"])
        xaxis = alt.Axis(values=list(range(0, MAXLAYERS + 100, 100)))
        xscale = alt.Scale(domain=(0, MAXLAYERS))
        _plot_linechart(
            df,
            "_img/deep_runtime",
            _nlayersEncoding,
            _mktitle("Running time, Deep circuits"),
            PL_L,
            axis=xaxis,
            scale=xscale,
        )
        _plot_trial_linechart(
            df,
            "_img/deep_runtime_trial",
            _nlayersEncoding,
            _mktitle("Running time/trial, Deep circuits"),
        )

        # Variational circuits
        vqeproblem = CATPROBLEMS["variational"]
        vqemeasure = "compile"
        def _diff_methods(problem) -> Iterable[Set[str]]:
            if a.plot_combine_adjoint_backprop:
                dmsame = set(["adjoint", "backprop"])
                return chain([set([x]) for x in set(DIFF_METHODS[(problem, vqemeasure)]) - dmsame], [dmsame])
            else:
                return [set([x]) for x in set(DIFF_METHODS[(problem, vqemeasure)])]

        df = _filter("variational", "compile", vqeproblem)
        for dms in _diff_methods(vqeproblem):
            df2 = df[df.get("diffmethod", pd.Series(float)).map(lambda m: m in dms)]
            if len(df2) > 0:
                dmtitle = "_".join(sorted(dms))
                fname = f"_img/variational_compile_{dmtitle.replace('-','')}_lineplot"
                _plot_linechart(
                    df2,
                    fname,
                    _nqubitsEncoding,
                    _mktitle(f"Compile time, Variational circuits ({dmtitle})"),
                )
                fname = f"_img/variational_compile_trial_{dmtitle.replace('-','')}_lineplot"
                _plot_trial_linechart(
                    df2,
                    fname,
                    _nqubitsEncoding,
                    _mktitle(f"Compile time/trial, Variational circuits ({dmtitle})"),
                )

        df = _filter("variational", "runtime", vqeproblem)
        for dms in _diff_methods(vqeproblem):
            df2 = df[df.get("diffmethod", pd.Series(float)).map(lambda m: m in dms)]
            if len(df2) > 0:
                dmtitle = "_".join(sorted(dms))
                fname = f"_img/variational_runtime_{dmtitle.replace('-','')}_lineplot"
                _plot_linechart(
                    df2,
                    fname,
                    _nqubitsEncoding,
                    _mktitle(f"Running time, Variational circuits ({dmtitle})"),
                )
                fname = f"_img/variational_runtime_trial_{dmtitle.replace('-','')}_lineplot"
                _plot_trial_linechart(
                    df2,
                    fname,
                    _nqubitsEncoding,
                    _mktitle(f"Running time/trial, Variational circuits ({dmtitle})"),
                )

        def _add_timeouts(df):
            for nqubits in sorted(set(df["nqubits"])):
                for impl in sorted(set(df["impl"])):
                    if len(df[(df["nqubits"] == nqubits) & (df["impl"] == impl)]) == 0:
                        df = pd.concat(
                            [
                                df,
                                DataFrame(
                                    [
                                        [
                                            impl,
                                            max(df["trial"]),
                                            nqubits,
                                            max(df[df["nqubits"] == nqubits]["timeout"]),
                                            max(df[df["nqubits"] == nqubits]["timeout"]),
                                        ]
                                    ],
                                    columns=["impl", "trial", "nqubits", "time", "timeout"],
                                ),
                            ]
                        )
            return df

        # Variational circuits, bar charts
        df_allgrad = _filter("variational", "runtime", vqeproblem)
        for diffmethods in _diff_methods(vqeproblem):
            df = df_allgrad[
                df_allgrad.get("diffmethod", pd.Series(float)).map(lambda m: m in diffmethods)
            ]
            if len(df) > 0:
                df = _add_timeouts(df)
                dmtitle = "_".join(sorted(diffmethods))
                fname = f"_img/variational_runtime_{dmtitle.replace('-','')}"
                writefile(
                    a,
                    fname,
                    alt.layer(
                        Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("impl", title=None, sort=list(ALIASES.values())),
                            y=alt.Y(
                                "mean(time):Q", title="Mean time, sec", scale=alt.Scale(type="log")
                            ),
                            color=alt.condition(
                                f"datum.mean_time >= {min(df_full['timeout'])}",
                                alt.ColorValue("Grey"),
                                _implEncoding(df, add_timeout=any(df_full["timeout"])),
                            ),
                        ),
                        Chart()
                        .mark_errorbar(extent="stderr")
                        .encode(
                            x=alt.X("impl", title=None, sort=list(ALIASES.values())),
                            y=alt.Y("time:Q", title="Mean time, sec"),
                        ),
                        data=df,
                    )
                    .facet(
                        column=alt.Column("nqubits:N", title="Qubits"),
                    )
                    .properties(
                        title=_mktitle(
                            f"Running time, Variational circuits ({dmtitle})", align=None
                        )
                    )
                    .to_dict(),
                )


# fmt: off
AP = ArgumentParser(prog="python3 batchrun.py")
AP.add_argument("-m", "--measure", type=str, default="all",
                help="Value to measure: compile|runtime|all, (default - 'all')")
AP.add_argument("-c", "--category", type=str, default="regular,deep,variational",
                help=("Category of circutis to evaluate regular|deep|hybrid|variational|all "
                "(default - 'regular,deep,variational')"))
AP.add_argument("--dry-run", default=False, action=BooleanOptionalAction,
                help="Enable this mode to print command lines but not actually run anything")
AP.add_argument("-a", "--actions", type=str, default="collect,plot",
                help="Which actions to perform (default - 'collect,plot')")
AP.add_argument("-t", "--timeout-1run", type=str, metavar="SEC", default="1000.0",
                help="Timeout for single benchmark run (default - 1000)")
AP.add_argument("--tag", type=str, default=None,
                help="Human-readable tag to add to the data file names")
AP.add_argument('-H', "--force-sysinfo-hash", type=str, default=None,
                help="Use the provided string instead of the system information hash")
AP.add_argument("--plot-formats", type=str, default="svg",
                help="Which formats to use for plotting ('svg[,png]', default - 'svg')")
AP.add_argument("--plot-combine-adjoint-backprop", default=False, action=BooleanOptionalAction,
                help="Plot adjoint and backprop diff. methods on the same plot")
AP.add_argument("-V", "--verbose", default=False, action=BooleanOptionalAction,
                help="Print verbose messages")
# fmt: on


def load_cmdline(cmdline: Optional[Union[str, list]] = None) -> DataFrame:
    """Loads all the dataframe"""
    if isinstance(cmdline, str):
        arglist = cmdline.split()
    elif isinstance(cmdline, list):
        arglist = cmdline
    elif cmdline is None:
        arglist = []
    else:
        raise ValueError(f"Unsupported {cmdline}")
    return load(AP.parse_args(arglist))


def load_tagged(syshash=SYSHASH, tag=f"v{FMTVERSION}") -> DataFrame:
    return load_cmdline(["-H", syshash, "--tag", tag])
