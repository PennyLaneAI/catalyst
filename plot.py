import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea

# data = np.load('peephole_benchmark_data_small.npz')
#data = np.load("peephole_benchmark_data_geomspace25.npz")
#data = np.load('timeit_peephole_benchmark_data.npz')
data = np.load('timeit_peephole_benchmark_data_geom25.npz')

loopsizes = data["loopsizes"] * 2  # each loop has 4 gates
walltimes = data["walltimes"]
cputimes = data["cputimes"]
programsizes = data["programsizes"]
core_PL_times = data["core_PL_times"]


print(loopsizes, walltimes, cputimes, programsizes, core_PL_times)


plt.figure(figsize=(15, 9))

# plt.plot(loopsizes, walltimes, label='wall time', color='green')
# plt.subplot(2, 1, 1)
plt.plot(
    loopsizes, cputimes, marker="o", label="Catalyst", c=plt.cm.tab20(2), ls="--", zorder=2
)
plt.plot(
    loopsizes, core_PL_times, marker="s", label="PennyLane", c=plt.cm.tab20(1), ls="--", zorder=2
)
# plt.title("Compilation time for running cancel_inverses and merge_rotations optimizations")
plt.xlabel("Circuit Gate Depth [$N$]", fontsize=14)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Compilation Time [ms]", fontsize=14)
# plt.ylim(0.29, 0.35)
plt.ylim(-100, 600000)
plt.legend(loc="upper right", fontsize=14)


img = mpimg.imread("auto_peephole_comp_horizontal.png")

# plt.xlim(-1000, 21000)
# plt.ylim(-100, 1500)
# plt.imshow(img, extent=(100, 1000, 600, 1000))

imagebox = OffsetImage(img, zoom=0.53)
ab = AnnotationBbox(imagebox, (300, 1270), zorder=1, frameon=False)
plt.gca().add_artist(ab)


text_box = TextArea(
    """
Compilation Time Benchmarks,
Quantum Circuit Optimizations

Catalyst v0.9.0-dev36
""",
    textprops=dict(color="black", fontsize=14),
)
ab = AnnotationBbox(
    text_box,
    (50, 180000),
    frameon=False,
    bboxprops=dict(facecolor="none", edgecolor="none"),
    zorder=1,
)
plt.gca().add_artist(ab)

"""
plt.subplot(2, 1, 2)
plt.plot(loopsizes, programsizes, label="program size", color="red")
plt.title("Catalyst compiled program size for programs with loops")
plt.xlabel("Circuit gate depth")
plt.ylabel("Program size (IR lines)")
plt.legend()


plt.subplot(3,1,3)
plt.plot(loopsizes, core_PL_times, label='core PL time', color='green')
plt.title('Plain PennyLane compilation time for running cancel_inverses and merge_rotations optimizations')
plt.xlabel('Circuit gate depth')
plt.ylabel('Compile time (ms)')
plt.legend()
"""

plt.subplots_adjust(hspace=0.8)
plt.show()
#plt.savefig("catalyst_quant_advantage_peephole_compile_time_artificial_circuit_log.png")
