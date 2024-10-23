import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea
import matplotlib.gridspec as gridspec

yellow = plt.cm.tab20(2)
blue = plt.cm.tab20(1)

# data = np.load('peephole_benchmark_data_small.npz')
#data = np.load("peephole_benchmark_data_geomspace25.npz")
#data = np.load('timeit_peephole_benchmark_data.npz')
#data = np.load('timeit_peephole_benchmark_data_geom25.npz')
data = np.load('timeit_peephole_benchmark_data_geom25_err.npz')

loopsizes = data["loopsizes"] * 2  # each loop has 4 gates
walltimes = data["walltimes"]
cputimes = data["cputimes"]
programsizes = data["programsizes"]
core_PL_times = data["core_PL_times"]
walltime_errs = data["walltime_errs"]
cputime_errs = data["cputime_errs"]
programsize_errs = data["programsize_errs"]
core_PL_time_errs = data["core_PL_time_errs"]


print(loopsizes, walltimes, cputimes, programsizes, core_PL_times,
    walltime_errs, cputime_errs, programsize_errs, core_PL_time_errs)



fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # 2:1 ratio
ax1 = fig.add_subplot(gs[0])  # First subplot (taller)
ax2 = fig.add_subplot(gs[1])  # Second subplot (shorter)

# plt.plot(loopsizes, walltimes, label='wall time', color='green')
#plt.subplot(2, 1, 1)
ax1.errorbar(
    loopsizes, cputimes, yerr=cputime_errs, marker="o", label="Catalyst", c=yellow, zorder=2
)
ax1.errorbar(
    loopsizes, core_PL_times, yerr=core_PL_time_errs, marker="s", label="PennyLane", c=blue, ls="--", zorder=2
)
# plt.title("Compilation time for running cancel_inverses and merge_rotations optimizations")
#plt.xlabel("Circuit Gate Depth [$N$]", fontsize=14)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel("Compilation Time [ms]", fontsize=14)
# plt.ylim(0.29, 0.35)
ax1.set_ylim(-0.00001, 700000)
ax1.legend(loc="upper right", fontsize=14, frameon=False)


img = mpimg.imread("auto_peephole_comp_horizontal.png")

# plt.xlim(-1000, 21000)
# plt.ylim(-100, 1500)
# plt.imshow(img, extent=(100, 1000, 600, 1000))

imagebox = OffsetImage(img, zoom=0.48)
ab = AnnotationBbox(imagebox, (1600, 9000), zorder=1, frameon=False)
ax1.add_artist(ab)
#plt.gca().add_artist(ab)


text_box = TextArea(
    """
$\mathbf{Compilation~Time~Benchmarks}$
$\mathbf{Quantum~Circuit~Optimizations}$
$\mathbf{Simple~Circuit}$

11th Gen Intel(R) Core(TM) i7-1185G7
Catalyst v0.9.0-dev36
""",
    textprops=dict(color="black", fontsize=14),
)
ab = AnnotationBbox(
    text_box,
    (50, 70000),
    frameon=False,
    bboxprops=dict(facecolor="none", edgecolor="none"),
    zorder=1,
)
ax1.add_artist(ab)
#plt.gca().add_artist(ab)
ax1.set_xticklabels([''] * len(ax1.get_xticks()))  # Remove the labels

#plt.subplot(2, 1, 2)
ax2.plot(loopsizes, core_PL_times/cputimes, c=yellow, marker="o")
#plt.title("Catalyst compiled program size for programs with loops")
ax2.set_xlabel("Circuit gate depth [N]")
ax2.set_ylabel("Compilation Speedup", fontsize=14)
ax2.set_xlabel("Circuit Gate Depth [$N$]", fontsize=14)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.grid(axis="y", zorder=0)
#plt.legend()

"""
plt.subplot(3,1,3)
plt.plot(loopsizes, core_PL_times, label='core PL time', color='green')
plt.title('Plain PennyLane compilation time for running cancel_inverses and merge_rotations optimizations')
plt.xlabel('Circuit gate depth')
plt.ylabel('Compile time (ms)')
plt.legend()
"""

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
#plt.show()
plt.savefig("catalyst_quant_advantage_peephole_compile_time_artificial_circuit_log_err.png", dpi=300)
