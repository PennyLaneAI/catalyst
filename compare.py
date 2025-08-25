import pennylane as qml
from catalyst import qjit
import numpy as np
import shutil, os, sys, tqdm
import pandas as pd
import random, subprocess

import matplotlib.pyplot as plt


def route_naively(num_qubits, coupling_map):
    run_python_program = f'''import pennylane as qml
dev = qml.device("null.qubit", wires = {coupling_map})
@qml.qjit()
@qml.qnode(dev)
def circuit():
    for i in range({num_qubits}):
        qml.H(i)
    for i in range({num_qubits}):
        for j in range(i):
            qml.CNOT([i,j])
    return qml.probs(0)
print(circuit())
'''
    result = subprocess.run(
        [sys.executable, "-c", run_python_program],
        capture_output=True,
        text=True,
        check=True
    )
    return (result.stdout.count("SWAP"))

def route_sabre(num_qubits, coupling_map_str):
    dev = qml.device("lightning.qubit")
    @qml.qjit
    @qml.qnode(dev)
    def circuit():
        for i in range(num_qubits):
            qml.H(i)
        for i in range(num_qubits):
            for j in range(i):
                qml.CNOT([i,j])
        return qml.state()
    
    temp_file_path = os.getcwd()+"/temp.mlir"
    f = open(temp_file_path,"w")
    f.writelines(circuit.mlir)
    f.close()
    run_command = os.getcwd() + '/mlir/build/bin/quantum-opt --route-circuit="hardware-graph={' + coupling_map_str + '}" ' + temp_file_path
    with os.popen(run_command) as p:
        output = p.read()
    return output.count("SWAP")



def run(num_physical_qubits, min_logical_qubit_count, max_logical_qubit_count, trials):
    df = []
    for num_logical_qubits in tqdm.tqdm(range(min_logical_qubit_count, max_logical_qubit_count)):
        row = {}
        row["num_qubits"] = num_logical_qubits
        swap_naive = []
        swap_sabre = []
        for _ in tqdm.tqdm(range(trials)):
            physical_qubits = list(range(num_physical_qubits))
            random.shuffle(physical_qubits)
            coupling_map = [(physical_qubits[i-1],physical_qubits[i]) for i in range(1,num_physical_qubits)]
            coupling_map_str = ''
            for item in coupling_map:
                coupling_map_str = coupling_map_str + "(" + str(item[0]) + "," + str(item[1]) + ");"
            swap_naive.append(route_naively(num_logical_qubits, coupling_map))
            swap_sabre.append(route_sabre(num_logical_qubits, coupling_map_str))
        print("Naive:", swap_naive)
        print("SABRE:", swap_sabre)
        row["SWAP_Naive"] = np.mean(swap_naive)
        row["SWAP_SABRE"] = np.mean(swap_sabre)
        row["SWAP_Naive_Dev"] = np.std(swap_naive)
        row["SWAP_SABRE_Dev"] = np.std(swap_sabre)
        df.append(row)
    df = pd.DataFrame(df)
    df.to_csv("comapre_results.csv",index=False)
    df = pd.read_csv("comapre_results.csv")

    plt.plot(
        df['num_qubits'],
        df['SWAP_SABRE'],
        marker='x', ls='--',label="SABRE"
    )
    plt.fill_between(
        df['num_qubits'],
        df['SWAP_SABRE'] - df["SWAP_SABRE_Dev"],
        df['SWAP_SABRE'] + df["SWAP_SABRE_Dev"],
        alpha=0.3,
    )

    plt.plot(
        df['num_qubits'],
        df['SWAP_Naive'],
        marker='x', ls='--',label="Naive"
    )
    plt.fill_between(
        df['num_qubits'],
        df['SWAP_Naive'] - df["SWAP_Naive_Dev"],
        df['SWAP_Naive'] + df["SWAP_Naive_Dev"],
        alpha=0.3,
    )

    plt.title("SWAP comparison for compiling QFT-like circuit to linear 15-qubit device")
    plt.xlabel("Number of qubits")
    plt.ylabel("SWAPs inserted")
    plt.legend()
    plt.grid()
    plt.savefig("comparison_plot.png",bbox_inches='tight')



trials = 5
num_physical_qubits = 50
min_logical_qubit_count = 5
max_logical_qubit_count = 11


# trials = 1
# num_physical_qubits = 100
# min_logical_qubit_count = 6
# max_logical_qubit_count = 101
run(num_physical_qubits, min_logical_qubit_count, max_logical_qubit_count, trials)
