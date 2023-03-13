""" Measurement cycle management entry point """
import sys

from catalyst_benchmark.toplevel import AP, load, collect, plot, SYSHASH_ORIG, SYSINFO, syshash

a = AP.parse_args(sys.argv[1:])
if a.verbose:
    print(f"Machine: {SYSINFO.toString()}\nHash {SYSHASH_ORIG}\nEffective {syshash(a)}")

if "collect" in a.actions:
    collect(a)
else:
    print("Skipping the 'collect' action")
if "plot" in a.actions:
    df, sysinfo = load(a)
    plot(a, df, sysinfo)
else:
    print("Skipping the 'plot' action")
