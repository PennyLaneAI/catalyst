import subprocess

import matplotlib.pyplot as plt
import numpy as np


def do(command):
    subprocess.call(command, shell=True)

def stderr(list_of_data):
    return np.std(list_of_data, ddof=1) / np.sqrt(len(list_of_data))

def run_one_circuit(timings, core_PL_timings, num_of_iters):
    do(f"python3 my_toy_circuit.py {num_of_iters}")

    with open("my_toy_circuit.yml", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "PeepholeBenchmarkPass" in line:
                walltime = lines[i + 1]
                walltime = float(walltime[walltime.find(":") + 2 :].strip("\n"))
                cputime = lines[i + 2]
                cputime = float(cputime[cputime.find(":") + 2 :].strip("\n"))
                programsize = lines[i + 3]
                programsize = int(programsize[programsize.find(":") + 2 :].strip("\n"))

                timings.append(
                    {"walltime": walltime, "cputime": cputime, "programsize": programsize}
                )

    with open("core_peephole_time.txt", "r") as f:
        core_PL_time = float(f.readlines()[0].strip("\n"))
        # print(core_PL_time)
        core_PL_timings.append(core_PL_time)

    return timings, core_PL_timings


def collect_mean(raw_data, core_PL_timings):
    """
    raw_data is something like
    [{'walltime': 0.307265, 'cputime': 0.306, 'programsize': 48},
     {'walltime': 0.301666, 'cputime': 0.299, 'programsize': 48},
     {'walltime': 0.342119, 'cputime': 0.341, 'programsize': 48},
     {'walltime': 0.313134, 'cputime': 0.312, 'programsize': 48}]

    core_PL_timings is just a list of numbers
    """
    mean_data = {"walltime": [], "cputime": [], "programsize": []}
    for i, d in enumerate(raw_data):
        mean_data["walltime"].append(d["walltime"])
        mean_data["cputime"].append(d["cputime"])
        mean_data["programsize"].append(d["programsize"])

    _ = mean_data["walltime"]
    mean_data["walltime"] = (np.mean(_), stderr(_))
    _ = mean_data["cputime"]
    mean_data["cputime"] = (np.mean(_), stderr(_))
    _ = mean_data["programsize"]
    mean_data["programsize"] = (np.mean(_), stderr(_))

    return mean_data, (np.mean(core_PL_timings), stderr(core_PL_timings))


def run_one_loopsize(loopsize):
    timings = []
    core_PL_timings = []
    for i in range(3):
        _ = run_one_circuit(timings, core_PL_timings, loopsize)
        timings = _[0]
        core_PL_timings = _[1]

    # print(timings)
    # print(collect_mean(timings))
    return collect_mean(timings, core_PL_timings)


############# main ##################
# loopsizes = [10, 50, 100, 500, 1000, 5000, 10000]
# loopsizes = [10, 20, 30, 40, 100, 150, 200]
loopsizes = np.geomspace(10, 50000, 25, dtype=int)
walltimes = []
cputimes = []
programsizes = []
core_PL_times = []

walltime_errs = []
cputime_errs = []
programsize_errs = []
core_PL_time_errs = []

for loopsize in loopsizes:
    _ = run_one_loopsize(loopsize)
    #breakpoint()
    core_PL_times.append(_[1][0])
    core_PL_time_errs.append(_[1][1])

    catalyst_times = _[0]
    walltimes.append(catalyst_times["walltime"][0])
    cputimes.append(catalyst_times["cputime"][0])
    programsizes.append(catalyst_times["programsize"][0])
    walltime_errs.append(catalyst_times["walltime"][0])
    cputime_errs.append(catalyst_times["cputime"][0])
    programsize_errs.append(catalyst_times["programsize"][0])


print(loopsizes, walltimes, cputimes, programsizes, core_PL_times)

loopsizes = np.array(loopsizes)
walltimes = np.array(walltimes)
cputimes = np.array(cputimes)
programsizes = np.array(programsizes)
core_PL_times = np.array(core_PL_times)
walltime_errs = np.array(walltime_errs)
cputime_errs = np.array(cputime_errs)
programsize_errs = np.array(programsize_errs)
core_PL_time_errs = np.array(core_PL_time_errs)

np.savez(
    "timeit_peephole_benchmark_data_geom25_err",
    loopsizes=loopsizes,
    walltimes=walltimes,
    cputimes=cputimes,
    programsizes=programsizes,
    core_PL_times=core_PL_times,
    walltime_errs=walltime_errs,
    cputime_errs=cputime_errs,
    programsize_errs=programsize_errs,
    core_PL_time_errs=core_PL_time_errs
)

do("rm -rf core_peephole_time.txt my_toy_circuit.yml")
