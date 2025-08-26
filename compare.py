import pennylane as qml
from catalyst import qjit
import numpy as np
import shutil, os, sys, tqdm
import pandas as pd
import random, subprocess

import matplotlib.pyplot as plt
import networkx as nx

###########################################################
######################   8x8 device   #####################
###########################################################
G = nx.grid_2d_graph(8,8)
G = nx.convert_node_labels_to_integers(G)

###########################################################
###############        Naive routing         ##############
###########################################################

# QFT string
def qft_string(num_qubits, coupling_map):
    qft_str = f'''import pennylane as qml
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
    return qft_str

# Route QFT naively
def route_qft_naively(num_qubits, coupling_map):
    run_python_program = qft_string(num_qubits, coupling_map)
    result = subprocess.run(
        [sys.executable, "-c", run_python_program],
        capture_output=True,
        text=True,
        check=True
    )
    return (result.stdout.count("SWAP"))

# Circular ansatz
def ansatz_string(num_qubits, coupling_map):
    ansatz_str = f'''import pennylane as qml
dev = qml.device("null.qubit", wires = {coupling_map})
@qml.qjit()
@qml.qnode(dev)
def circuit():
    for i in range({num_qubits}):
        qml.H(i)
    for i in range(1,{num_qubits}):
        qml.CNOT([i-1,i])
    
    for i in range({num_qubits}):
        qml.H(i)
    for i in range(1,{num_qubits}):
        qml.CNOT([i-1,i])
    
    for i in range({num_qubits}):
        qml.H(i)
    return qml.probs(0)
print(circuit())
'''
    return ansatz_str

# Circular ansatz route naively
def route_ansatz_naively(num_qubits, coupling_map):
    run_python_program = ansatz_string(num_qubits, coupling_map)
    result = subprocess.run(
        [sys.executable, "-c", run_python_program],
        capture_output=True,
        text=True,
        check=True
    )
    return (result.stdout.count("SWAP"))

###########################################################
###############        SABRE routing         ##############
###########################################################

# Ansatz circuit
def ansatz_circuit(num_qubits):
    for i in range(num_qubits):
        qml.H(i)
    for i in range(1,num_qubits):
        qml.CNOT([i-1,i])
    
    for i in range(num_qubits):
        qml.H(i)
    for i in range(1,num_qubits):
        qml.CNOT([i-1,i])

    for i in range(num_qubits):
        qml.H(i)
    
# route Ansatz using SABRE
def route_ansatz_sabre(num_qubits, coupling_map_str):
    dev = qml.device("lightning.qubit")
    @qml.qjit
    @qml.qnode(dev)
    def circuit():
        ansatz_circuit(num_qubits)
        return qml.state()
    
    temp_file_path = os.getcwd()+"/temp.mlir"
    f = open(temp_file_path,"w")
    f.writelines(circuit.mlir)
    f.close()
    run_command = os.getcwd() + '/mlir/build/bin/quantum-opt --route-circuit="hardware-graph={' + coupling_map_str + '}" ' + temp_file_path
    with os.popen(run_command) as p:
        output = p.read()
    return output.count("SWAP")

# QFT circuit
def qft_circuit(num_qubits):
    for i in range(num_qubits):
        qml.H(i)
    for i in range(num_qubits):
        for j in range(i):
            qml.CNOT([i,j])
    
# route QFT using SABRE
def route_qft_sabre(num_qubits, coupling_map_str):
    dev = qml.device("lightning.qubit")
    @qml.qjit
    @qml.qnode(dev)
    def circuit():
        qft_circuit(num_qubits)
        return qml.state()
    
    temp_file_path = os.getcwd()+"/temp.mlir"
    f = open(temp_file_path,"w")
    f.writelines(circuit.mlir)
    f.close()
    run_command = os.getcwd() + '/mlir/build/bin/quantum-opt --route-circuit="hardware-graph={' + coupling_map_str + '}" ' + temp_file_path
    with os.popen(run_command) as p:
        output = p.read()
    return output.count("SWAP")

def post_process(filename, circuit_name):
    df = pd.read_csv(filename+".csv")
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

    plt.title("SWAP comparison for "+ circuit_name +"-like circuit to 10x10 grid device")
    plt.xlabel("Number of qubits")
    plt.ylabel("SWAPs inserted")
    plt.legend()
    plt.grid()
    plt.savefig(filename+".png",bbox_inches='tight')
    plt.clf()


def run_sabre(min_logical_qubit_count, max_logical_qubit_count, trials):
    df = []
    for num_logical_qubits in tqdm.tqdm(range(min_logical_qubit_count, max_logical_qubit_count)):
        row = {}
        row["num_qubits"] = num_logical_qubits
        swap_naive = []
        swap_sabre = []
        for _ in tqdm.tqdm(range(trials)):
            original_nodes = list(G.nodes())
            random.shuffle(original_nodes)
            H = nx.relabel_nodes(G, {i:original_nodes[i] for i in range(len(original_nodes))})
            coupling_map = list(H.edges())
            coupling_map_str = ''
            for item in coupling_map:
                coupling_map_str = coupling_map_str + "(" + str(item[0]) + "," + str(item[1]) + ");"
            swap_naive.append(route_qft_naively(num_logical_qubits, coupling_map))
            swap_sabre.append(route_qft_sabre(num_logical_qubits, coupling_map_str))
        print("Naive:", swap_naive)
        print("SABRE:", swap_sabre)
        row["SWAP_Naive"] = np.mean(swap_naive)
        row["SWAP_SABRE"] = np.mean(swap_sabre)
        row["SWAP_Naive_Dev"] = np.std(swap_naive)
        row["SWAP_SABRE_Dev"] = np.std(swap_sabre)
        df.append(row)
    df = pd.DataFrame(df)
    df.to_csv("comapre_results_qft.csv",index=False)
    post_process("comapre_results_qft", "QFT")


def run_ansatz(min_logical_qubit_count, max_logical_qubit_count, trials):
    df = []
    for num_logical_qubits in tqdm.tqdm(range(min_logical_qubit_count, max_logical_qubit_count)):
        row = {}
        row["num_qubits"] = num_logical_qubits
        swap_naive = []
        swap_sabre = []
        for _ in tqdm.tqdm(range(trials)):
            original_nodes = list(G.nodes())
            random.shuffle(original_nodes)
            H = nx.relabel_nodes(G, {i:original_nodes[i] for i in range(len(original_nodes))})
            coupling_map = list(H.edges())
            coupling_map_str = ''
            for item in coupling_map:
                coupling_map_str = coupling_map_str + "(" + str(item[0]) + "," + str(item[1]) + ");"
            swap_naive.append(route_ansatz_naively(num_logical_qubits, coupling_map))
            swap_sabre.append(route_ansatz_sabre(num_logical_qubits, coupling_map_str))
        print("Naive:", swap_naive)
        print("SABRE:", swap_sabre)
        row["SWAP_Naive"] = np.mean(swap_naive)
        row["SWAP_SABRE"] = np.mean(swap_sabre)
        row["SWAP_Naive_Dev"] = np.std(swap_naive)
        row["SWAP_SABRE_Dev"] = np.std(swap_sabre)
        df.append(row)
    df = pd.DataFrame(df)
    df.to_csv("comapre_results_ansatz.csv",index=False)
    post_process("comapre_results_ansatz", "Ansatz")


# trials = 5
# min_logical_qubit_count = 2
# max_logical_qubit_count = 11


trials = 50
min_logical_qubit_count = 2
max_logical_qubit_count = 25
run_sabre(min_logical_qubit_count, max_logical_qubit_count, trials)
run_ansatz(min_logical_qubit_count, max_logical_qubit_count, trials)
