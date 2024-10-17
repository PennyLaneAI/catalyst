import subprocess

import matplotlib.pyplot as plt
import numpy as np


def do(command):
    subprocess.call(command, shell=True)


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
    mean_data = {"walltime": 0, "cputime": 0, "programsize": 0}
    for i, d in enumerate(raw_data):
        mean_data["walltime"] += d["walltime"]
        mean_data["cputime"] += d["cputime"]
        mean_data["programsize"] += d["programsize"]

    N = len(raw_data)
    mean_data["walltime"] /= N
    mean_data["cputime"] /= N
    mean_data["programsize"] //= N

    return mean_data, np.mean(np.array(core_PL_timings))


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

for loopsize in loopsizes:
    _ = run_one_loopsize(loopsize)
    catalyst_times = _[0]
    core_PL_times.append(_[1])
    walltimes.append(catalyst_times["walltime"])
    cputimes.append(catalyst_times["cputime"])
    programsizes.append(catalyst_times["programsize"])

print(loopsizes, walltimes, cputimes, programsizes, core_PL_times)

loopsizes = np.array(loopsizes)
walltimes = np.array(walltimes)
cputimes = np.array(cputimes)
programsizes = np.array(programsizes)
core_PL_times = np.array(core_PL_times)

np.savez(
    "peephole_benchmark_data",
    loopsizes=loopsizes,
    walltimes=walltimes,
    cputimes=cputimes,
    programsizes=programsizes,
    core_PL_times=core_PL_times,
)


#################### plot ########################
plot = False
if plot:
    plt.figure(figsize=(18, 10))

    # plt.plot(loopsizes, walltimes, label='wall time', color='green')
    plt.subplot(2, 1, 1)
    plt.plot(loopsizes, cputimes, label="Catalyst time", color="blue")
    plt.plot(loopsizes, core_PL_times, label="core PL time", color="green")
    plt.title("Compilation time for running cancel_inverses and merge_rotations optimizations")
    plt.xlabel("Circuit gate depth")
    plt.ylabel("Compile time (ms)")
    # plt.ylim(0.29, 0.35)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loopsizes, programsizes, label="program size", color="red")
    plt.title("Catalyst compiled program size for programs with loops")
    plt.xlabel("Circuit gate depth")
    plt.ylabel("Program size (IR lines)")
    plt.legend()

    """
	plt.subplot(3,1,3)
	plt.plot(loopsizes, core_PL_times, label='core PL time', color='green')
	plt.title('Plain PennyLane compilation time for running cancel_inverses and merge_rotations optimizations')
	plt.xlabel('Circuit gate depth')
	plt.ylabel('Compile time (ms)')
	plt.legend()
	"""

    plt.subplots_adjust(hspace=0.8)
    plt.show()
