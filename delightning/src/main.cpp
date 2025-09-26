
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>
#include <limits>
#include <queue>
#include <algorithm>
#include <regex>
#include <unordered_set>


// Operator
// ______________________________
// * name<string>: Operator name
// + getName(): string

class Operator {
    // TODO: string_view
    std::string name;

public:
    Operator() = default;
    explicit Operator(const std::string& name) : name(name) {}
    std::string getName() const { return name; }

    bool operator==(const Operator& other) const {
        return name == other.name;
    }
    bool operator!=(const Operator& other) const {
        return !(*this == other);
    }
};

namespace std {
    template <>
    struct hash<Operator> {
        std::size_t operator()(const Operator& op) const noexcept {
            return std::hash<std::string>()(op.getName());
        }
    };
}




// ResourceOp
// ______________________________
// * resources<umap<Operator, Int>>: Resources
// + getResources(): umap
// + total_cost(): int
// + op_cost(Operator): int
// + has_op(Operator): bool

class ResourceOp {
    std::unordered_map<Operator, int> resources;
    size_t total = 0;

public:
    ResourceOp() = default;
    explicit ResourceOp(const std::unordered_map<Operator, int>& resources) : resources(resources) {}

    const std::unordered_map<Operator, int>& getResources() const {
        return resources;
    }

    size_t total_cost() {
        if (total == 0 && !resources.empty()) {
            for (const auto& pair : resources) {
                total += pair.second;
            }
        }
        return total;
    }

    size_t op_cost(const Operator& op) const {
        auto it = resources.find(op);
        return (it != resources.end()) ? it->second : 0;
    }

    bool has_op(const Operator& op) const {
        return resources.find(op) != resources.end();
    }
};



// RuleRefOp
// _________________________________________
// * Op<Operator>: Operator
// * Resource<ResourceOp>: Resources
// * RuleRef<string>: Pointer to the rule

class RuleRefOp {
    Operator op;
    ResourceOp resources;
    std::string rule_ref;

public:
    RuleRefOp(const Operator& op, const ResourceOp& resources, const std::string& rule_ref)
        : op(op), resources(resources), rule_ref(rule_ref) {}

    const Operator& getOperator() const { return op; }
    const std::string& getRuleRef() const { return rule_ref; }

    // TODO: make this const ref to avoid copy overhead
    // this is currently required for my simple Dijkstra sovler
    ResourceOp getResources() const { return resources; }
};



// BasicSolver <- experimental!
// Check PLSolver following PL's implementation
// _________________________________________
// * Ops<vector<Operator>>: Operators
// * Gateset<vector<Operator>>: Operators
// * Rules<vector<RuleRefOp>>: Rules
// + graph(): void
// + show(): stdout
// + solve(): map<Operator, RuleRef>

class BasicSolver {
private:
    std::vector<Operator> ops;
    std::vector<Operator> gateset;
    std::vector<RuleRefOp> rules;

    // Cached solutions and distances maps
    std::unordered_map<Operator, std::string> solutions;
    std::unordered_map<Operator, size_t> distances;

    size_t computeCost(const RuleRefOp& rule) const {
        size_t total = 0;
        // std::cerr << "[DEBUG]     Computing cost for rule: " << rule.getRuleRef() << "\n";
        for (const auto& [dep, count] : rule.getResources().getResources()) {
            auto it = distances.find(dep);
            // std::cerr << "[DEBUG]       Dependency: " << dep.getName()
            //           << " with count: " << count
            //           << " and cost: " << it->second << "\n";
            if (it == distances.end() || it->second == std::numeric_limits<size_t>::max()) {
                // Dependency not found or unreachable yet :(
                return std::numeric_limits<size_t>::max();
            }
            total += count * it->second;
        }
        return total;
    }

    auto initGraph() {
        using NodeOp = std::pair<size_t, Operator>;
        auto cmp = [](const NodeOp& left, const NodeOp& right) {
            return left.first > right.first;
        };
        std::priority_queue<NodeOp, std::vector<NodeOp>, decltype(cmp)> queue(cmp);

        for (const auto& op : ops) {
            distances[op] = std::numeric_limits<size_t>::max();
            queue.push({distances[op], op});
        }

        for (const auto& g : gateset) {
            distances[g] = 1;
            queue.push({1, g});
            solutions[g] = "base_op";
        }
        return queue;
    }

public:
    BasicSolver(const std::vector<Operator>& ops,
                const std::vector<Operator>& gateset,
                const std::vector<RuleRefOp>& rules)
        : ops(ops), gateset(gateset), rules(rules) {}

    bool isBasisGate(const Operator& op) const {
        return std::find(gateset.begin(), gateset.end(), op) != gateset.end();
    }

    std::unordered_map<Operator, std::string> solve() {
        auto queue = initGraph();

        while (!queue.empty()) {
            auto [current_distance, current_op] = queue.top();
            queue.pop();

            // If we found a better path, skip processing
            if (current_distance > distances[current_op]) {
                continue;
            }

            // std::cerr << "[DEBUG] Exploring neighbors of operator: "
            //     << current_op.getName() << "\n";

            // Explore neighbors :)
            for (const auto& rule : rules) {
                // std::cerr << "[DEBUG] Considering rule: " << rule.getRuleRef()
                //           << " for operator: " << rule.getOperator().getName()
                //           << " with total cost: " << rule.getResources().total_cost()
                //           << "\n";
                if (rule.getOperator() != current_op) {
                    continue;
                }
                
                // std::cerr << "[DEBUG] Found applicable rule: " << rule.getRuleRef()
                //           << " for operator: " << current_op.getName() << "\n";
                size_t new_distance = computeCost(rule);

                // std::cerr << "[DEBUG] New computed distance for operator: "
                //           << current_op.getName() << " is " << new_distance << "\n";

                if (new_distance < distances[current_op]) {
                    distances[current_op] = new_distance;
                    queue.push({new_distance, current_op});
                    solutions[current_op] = rule.getRuleRef();
                    // std::cerr << "[DEBUG] Updating distance for operator: "
                    //           << current_op.getName() << " to " << new_distance << "\n";
                }
            }
        }

        return solutions;
    }

    // For testing purposes (my first try)
    std::unordered_map<Operator, std::string> simple_solver() {
        if (!solutions.empty()) {
            return solutions;
        }

        // We need to create a distance map for our Dijkstra's algorithm
        // For now, I keep everything simple starting with max distance
        // TODO: do this part implicitly for performance
        std::unordered_map<Operator, size_t> distances;
        for (const auto& op : ops) {
            distances[op] = std::numeric_limits<size_t>::max();
        }

        // There are different ways to implement Dijkstra's algorithm
        // Here, I use a simple priority queue for demonstration
        // TODO: optimize with a better priority queue or min-heap
        using QElement = std::pair<size_t, Operator>; // (distance, operator)
        auto cmp = [](const QElement& left, const QElement& right) { return left.first > right.first; };
        std::priority_queue<QElement, std::vector<QElement>, decltype(cmp)> queue(cmp);

        // Initialize the queue with all operators and distance 0
        for (const auto& op : ops) {
            queue.push({0, op});
        }

        // Dijkstra's algorithm main loop
        while (!queue.empty()) {
            auto [current_distance, current_op] = queue.top();
            queue.pop();
            // If we found a better path, skip processing
            if (current_distance > distances[current_op]) {
                continue;
            }

            // std::cerr << "[DEBUG] Exploring neighbors of operator: "
            //     << current_op.getName() << "\n";

            // Explore neighbors :)
            for (const auto& rule: rules) {
                if (rule.getOperator() == current_op) {
                    // std::cerr << "[DEBUG] Found applicable rule: " << rule.getRuleRef()
                    //         << " for operator: " << current_op.getName()
                    //         << " with total cost: " << rule.getResources().total_cost()
                    //         << "\n";
                    size_t new_distance = current_distance + rule.getResources().total_cost();
                    if (new_distance < distances[current_op]) {
                        // std::cerr << "[DEBUG] Updating distance for operator: " << current_op.getName()
                        //           << " from " << distances[current_op]
                        //           << " to " << new_distance << "\n";
                        distances[current_op] = new_distance;
                        queue.push({new_distance, current_op});

                        // Update solution
                        solutions[current_op] = rule.getRuleRef();
                    }
                }
            }
        }

        return solutions;
    }


    void show() {
        for (const auto& [op, rule] : solutions) {
            std::cout << "Operator " << op.getName()
                      << " decomposed using rule: " << rule << "\n";
        }
    }
};


// PLSolver w/ Operator and Rule Nodes 

enum class NodeType {
    OPERATOR,
    RULE
};

struct Node {
    NodeType type;
    Operator op;
    RuleRefOp rule;
    size_t index;
};

struct Edge {
    size_t target;
    size_t weight;
};

class Graph {
private:
    std::vector<Node> nodes;
    std::vector<std::vector<Edge>> adjList;

public:
    Graph() = default;

    size_t addNode(const Node& node) {
        const size_t idx = nodes.size();
        nodes.push_back(node);
        adjList.emplace_back();
        return idx;
    }

    void addEdge(size_t from, size_t to, size_t weight) {
        adjList[from].push_back({to, weight});
    }

    const Node& getNode(size_t index) const {
        return nodes[index];
    }

    size_t size() const {
        return nodes.size();
    }

    const std::vector<Edge>& getNeighbors(size_t index) const {
        return adjList[index];
    }
};


Graph buildGraph(
    const std::vector<Operator>& ops,
    const std::vector<Operator>& gateset,
    const std::vector<RuleRefOp>& rules)
{
    Graph graph;
    std::unordered_map<Operator, size_t> opNodes;

    // Create Operator nodes
    for (const auto& op: ops) {
        size_t idx = graph.addNode({NodeType::OPERATOR, op, RuleRefOp(op, {}, ""), 0});
        opNodes[op] = idx;
    }

    for (const auto &op: gateset) {
        size_t idx = graph.addNode({NodeType::OPERATOR, op, RuleRefOp(op, {}, ""), 0});
        opNodes[op] = idx;
    }

    // Create Rule nodes and edges
    for (const auto& rule: rules) {
        size_t ruleIdx = graph.addNode({NodeType::RULE, {}, rule, 0});
        auto op = rule.getOperator();
        size_t opIdx = opNodes[op];

        // Op -> Rule edge
        graph.addEdge(opIdx, ruleIdx, 0);

        // Rule -> deps edges
        for (const auto &[dep, count] : rule.getResources().getResources()) {
            if (!opNodes.count(dep)) {
                size_t depIdx = graph.addNode({NodeType::OPERATOR, dep, RuleRefOp(dep, {}, ""), 0});
                opNodes[dep] = depIdx;
            }
            graph.addEdge(ruleIdx, opNodes[dep], count);
        }

    }

    return graph;
}


std::unordered_map<Operator, std::string>
solveGraph(Graph& graph) {
    using ElemPair = std::pair<size_t, size_t>; // (distance, nodeIndex)
    auto cmp = [](const ElemPair& a, const ElemPair& b) { return a.first > b.first; };
    std::priority_queue<ElemPair, std::vector<ElemPair>, decltype(cmp)> queue(cmp);

    std::vector<size_t> dist(graph.size(), std::numeric_limits<size_t>::max());
    std::unordered_map<Operator, std::string> solutions;

    // Start with gateset operators = cost 0
    for (size_t i = 0; i < graph.size(); i++) {
        auto& node = graph.getNode(i);
        if (node.type == NodeType::OPERATOR && dist[i] == std::numeric_limits<size_t>::max()) {
            // Basis gate → distance 0
            if (solutions.count(node.op) == 0) {
                dist[i] = 0;
                queue.push({0, i});
            }
        }
    }

    while (!queue.empty()) {
        auto [curDist, u] = queue.top();
        queue.pop();

        if (curDist > dist[u]) continue;

        auto& uNode = graph.getNode(u);

        // Explore neighbors
        for (auto& edge : graph.getNeighbors(u)) {
            auto& vNode = graph.getNode(edge.target);

            size_t newDist = 0;
            if (uNode.type == NodeType::OPERATOR && vNode.type == NodeType::RULE) {
                // Operator → Rule: defer cost to expansion
                newDist = curDist;
            } else if (uNode.type == NodeType::RULE && vNode.type == NodeType::OPERATOR) {
                // Rule → Operator: accumulate resource counts
                size_t count = uNode.rule.getResources().op_cost(vNode.op);
                newDist = curDist + count * dist[edge.target];
            } else {
                continue;
            }

            if (newDist < dist[edge.target]) {
                dist[edge.target] = newDist;
                queue.push({newDist, edge.target});

                // If we reached an operator from a rule, record the chosen rule
                if (vNode.type == NodeType::OPERATOR && uNode.type == NodeType::RULE) {
                    solutions[vNode.op] = uNode.rule.getRuleRef();
                }
            }
        }
    }

    return solutions;
}

// ----------------------------
// MLIR Parser for quantum.custom ops
// ----------------------------

auto parse_quantum_custom_ops(const std::string& mlir_code) {
    std::unordered_set<Operator> ops;

    std::regex pattern(R"(quantum\.custom\s+\"([A-Za-z0-9_\.]+)\")");

    std::smatch matches;
    std::string::const_iterator search_start(mlir_code.cbegin());
    while (std::regex_search(search_start, mlir_code.cend(), matches, pattern)) {
        ops.emplace(matches[1].str());
        search_start = matches.suffix().first;
    }

    return ops;
}

// ----------------------------
// Simple Tests
// ----------------------------

void test_operator() {
    Operator op1("H");
    Operator op2("X");
    Operator op3("H");

    assert(op1.getName() == "H");
    assert(op2.getName() == "X");
    assert(!(op1 == op2));
    assert(op1 != op2);
    assert(op1 == op3);

    std::cout << "[PASS] Operator tests" << std::endl;
}

void test_resourceop() {
    Operator op1("H");
    Operator op2("X");

    std::unordered_map<Operator, int> res{{op1, 3}, {op2, 5}};
    ResourceOp r(res);

    assert(r.total_cost() == 8);
    assert(r.op_cost(op1) == 3);
    assert(r.op_cost(op2) == 5);
    assert(r.op_cost(Operator("Z")) == 0);
    assert(r.has_op(op1));
    assert(!r.has_op(Operator("Z")));

    std::cout << "[PASS] ResourceOp tests" << std::endl;
}


void test_rulerefop() {
    Operator op("CX");
    ResourceOp r({{op, 2}});
    RuleRefOp rr(op, r, "rule1");

    assert(rr.getOperator() == op);
    assert(rr.getRuleRef() == "rule1");
    assert(rr.getResources().op_cost(op) == 2);

    std::cout << "[PASS] RuleRefOp tests" << std::endl;
}

void test_solver1() {

    Operator cnot("CNOT");
    Operator cz("CZ");
    Operator h("H");

    ResourceOp cz_to_cnot({{cnot, 1}, {h, 2}});
    ResourceOp h_self({{h, 1}});

    RuleRefOp rule1(cz, cz_to_cnot, "cz_decomp_rule");
    RuleRefOp rule2(h, h_self, "h_rule");

    std::vector<Operator> ops = {cz, h};
    std::vector<Operator> gateset = {cnot, h};
    std::vector<RuleRefOp> rules = {rule1, rule2};

    Solver solver(ops, gateset, rules);
    auto solutions = solver.solve();
    // solver.show();
    assert(solutions.size() == 3);
    assert(solutions[cz] == "cz_decomp_rule");
    assert(solutions[h] == "h_rule");

    std::cout << "[PASS] Solver tests (1)" << std::endl;
}

void test_solver2() {

    Operator cz("CZ");
    Operator cnot("CNOT");
    Operator h("H");
    Operator rz("RZ");
    Operator rx("RX");

    ResourceOp cz_to_h_cnot({{h, 1}, {cnot, 1}});
    RuleRefOp rule1(cz, cz_to_h_cnot, "cz_h_cnot_rule");

    ResourceOp cz_to_rx_rz_cnot({{rx, 1}, {rz, 1}, {cnot, 1}});
    RuleRefOp rule2(cz, cz_to_rx_rz_cnot, "cz_rx_rz_cnot_rule");

    ResourceOp h_to_rz_rz({{rz, 2}});
    RuleRefOp rule3(h, h_to_rz_rz, "h_rz_rz_rule");

    ResourceOp h_to_rz_rx_rz({{rz, 2}, {rx, 1}});
    RuleRefOp rule4(h, h_to_rz_rx_rz, "h_rz_rx_rz_rule");

    std::vector<Operator> ops = {h, cz};
    std::vector<Operator> gateset = {cnot, rz, rx};
    std::vector<RuleRefOp> rules = {rule1, rule2, rule3, rule4};

    Solver solver(ops, gateset, rules);
    auto solutions = solver.solve();
    // solver.show();
    assert(solutions.size() == 5);
    assert(solutions[cz] == "cz_h_cnot_rule");
    assert(solutions[h] == "h_rz_rz_rule");
    assert(solutions[rz] == "base_op");
    assert(solutions[rx] == "base_op");
    assert(solutions[cnot] == "base_op");

    std::cout << "[PASS] Solver tests (2)" << std::endl;
}


void test_solver3() {
    // Define Operators
    Operator single_exc("SingleExcitation");
    Operator single_exc_plus("SingleExcitationPlus");
    Operator double_exc("DoubleExcitation");
    Operator cry("CRY");
    Operator s("S");
    Operator phase("PhaseShift");
    Operator rz("RZ");
    Operator rx("RX");
    Operator ry("RY");
    Operator rot("Rot");
    Operator hadamard("Hadamard");
    Operator cnot("CNOT");
    Operator cy("CY");
    Operator t("T");
    Operator global_phase("GlobalPhase");
    Operator phaseshift("PhaseShift");

    // ('SingleExcitation', {H:2, CNOT:2, RY:2}, _single_excitation_decomp)
    ResourceOp res_single_exc({{hadamard, 2}, {cnot, 2}, {ry, 2}});
    RuleRefOp rule_single_exc(single_exc, res_single_exc, "_single_excitation_decomp");

    // ('SingleExcitationPlus', {H:2, CY:1, CNOT:2, RY:2, S:1, RZ:1, GlobalPhase:1}, _single_excitation_plus_decomp)
    ResourceOp res_single_exc_plus({
        {hadamard, 2}, {cy, 1}, {cnot, 2}, {ry, 2},
        {s, 1}, {rz, 1}, {global_phase, 1}});
    RuleRefOp rule_single_exc_plus(single_exc_plus, res_single_exc_plus, "_single_excitation_plus_decomp");

    // ('DoubleExcitation', {CNOT:14, H:6, RY:8}, _doublexcit)
    ResourceOp res_double_exc1({{cnot, 14}, {hadamard, 6}, {ry, 8}});
    RuleRefOp rule_double_exc1(double_exc, res_double_exc1, "_doublexcit");

    // ('CRY', {RY:2, CNOT:2}, _cry)
    ResourceOp res_cry({{ry, 2}, {cnot, 2}});
    RuleRefOp rule_cry(cry, res_cry, "_cry");

    // ('S', {PhaseShift:1}, _s_phaseshift)
    ResourceOp res_s1({{phase, 1}});
    RuleRefOp rule_s1(s, res_s1, "_s_phaseshift");

    // ('S', {T:1}, _s_to_t)
    ResourceOp res_s2({{t, 1}});
    RuleRefOp rule_s2(s, res_s2, "_s_to_t");

    // ('PhaseShift', {RZ:1, GlobalPhase:1}, _phaseshift_to_rz_gp)
    ResourceOp res_phase({{rz, 1}, {global_phase, 1}});
    RuleRefOp rule_phase(phase, res_phase, "_phaseshift_to_rz_gp");

    // ('RZ', {Rot:1}, _rz_to_rot)
    ResourceOp res_rz1({{rot, 1}});
    RuleRefOp rule_rz1(rz, res_rz1, "_rz_to_rot");

    // ('RZ', {RY:2, RX:1}, _rz_to_ry_rx)
    ResourceOp res_rz2({{ry, 2}, {rx, 1}});
    RuleRefOp rule_rz2(rz, res_rz2, "_rz_to_ry_rx");

    // ('Rot', {RZ:2, RY:1}, _rot_to_rz_ry_rz)
    ResourceOp res_rot({{rz, 2}, {ry, 1}});
    RuleRefOp rule_rot(rot, res_rot, "_rot_to_rz_ry_rz");


    std::vector<Operator> ops = {single_exc, single_exc_plus, double_exc};
    std::vector<Operator> gateset = {ry, rx, cnot, hadamard, global_phase};
    std::vector<RuleRefOp> rules = {
        rule_single_exc, rule_single_exc_plus,
        rule_double_exc1,
        rule_cry, rule_s1, rule_s2,
        rule_phase, rule_rz1, rule_rz2,
        rule_rot
    };

    Solver solver(ops, gateset, rules);
    auto solutions = solver.solve();
    // solver.show();
    assert(solutions.size() == 8);
    assert(solutions[single_exc] == "_single_excitation_decomp");
    assert(solutions[single_exc_plus] == "_single_excitation_plus_decomp");
    assert(solutions[double_exc] == "_doublexcit");
    assert(solutions[ry] == "base_op");
    assert(solutions[rx] == "base_op");
    assert(solutions[cnot] == "base_op");
    assert(solutions[hadamard] == "base_op");
    assert(solutions[global_phase] == "base_op");

    std::cout << "[PASS] Solver tests (3)" << std::endl;
}

void test_solver4() {
    std::string mlir_code = R"(
        func.func public @circuit_15() -> tensor<f64> attributes {decompose_gatesets = [["GlobalPhase", "RY", "Hadamard", "CNOT", "RX"]], diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
              %cst = arith.constant 1.250000e-01 : f64
              %cst_0 = arith.constant -1.250000e-01 : f64
              %cst_1 = arith.constant -2.500000e-01 : f64
              %cst_2 = arith.constant 2.500000e-01 : f64
              %cst_3 = arith.constant 5.000000e-01 : f64
              %c0_i64 = arith.constant 0 : i64
              quantum.device shots(%c0_i64) ["/home/ali/miniforge3/envs/decomp/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
              %0 = quantum.alloc( 4) : !quantum.reg
              %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
              %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
              %out_qubits:2 = quantum.custom "SingleExcitation"(%cst_3) %1, %2 : !quantum.bit, !quantum.bit
              %out_qubits_4 = quantum.custom "Hadamard"() %out_qubits#1 : !quantum.bit
              %out_qubits_5:2 = quantum.custom "CNOT"() %out_qubits_4, %out_qubits#0 : !quantum.bit, !quantum.bit
              %out_qubits_6 = quantum.custom "RY"(%cst_2) %out_qubits_5#1 : !quantum.bit
              %out_qubits_7 = quantum.custom "RY"(%cst_2) %out_qubits_5#0 : !quantum.bit
              %out_qubits_8:2 = quantum.custom "CY"() %out_qubits_7, %out_qubits_6 : !quantum.bit, !quantum.bit
              %out_qubits_9 = quantum.custom "S"() %out_qubits_8#0 : !quantum.bit
              %out_qubits_10 = quantum.custom "Hadamard"() %out_qubits_9 : !quantum.bit
              %out_qubits_11 = quantum.custom "RZ"(%cst_1) %out_qubits_10 : !quantum.bit
              %out_qubits_12:2 = quantum.custom "CNOT"() %out_qubits_8#1, %out_qubits_11 : !quantum.bit, !quantum.bit
              quantum.gphase(%cst_0) :
              %out_qubits_13 = quantum.custom "Hadamard"() %out_qubits_12#1 : !quantum.bit
              %out_qubits_14:2 = quantum.custom "CNOT"() %out_qubits_13, %out_qubits_12#0 : !quantum.bit, !quantum.bit
              %out_qubits_15 = quantum.custom "RY"(%cst_2) %out_qubits_14#1 : !quantum.bit
              %out_qubits_16 = quantum.custom "RY"(%cst_2) %out_qubits_14#0 : !quantum.bit
              %out_qubits_17:2 = quantum.custom "CY"() %out_qubits_16, %out_qubits_15 : !quantum.bit, !quantum.bit
              %out_qubits_18 = quantum.custom "S"() %out_qubits_17#0 : !quantum.bit
              %out_qubits_19 = quantum.custom "Hadamard"() %out_qubits_18 : !quantum.bit
              %out_qubits_20 = quantum.custom "RZ"(%cst_2) %out_qubits_19 : !quantum.bit
              %out_qubits_21:2 = quantum.custom "CNOT"() %out_qubits_17#1, %out_qubits_20 : !quantum.bit, !quantum.bit
              quantum.gphase(%cst) :
              %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
              %4 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit
              %out_qubits_22:4 = quantum.custom "DoubleExcitation"(%cst_3) %out_qubits_21#0, %out_qubits_21#1, %3, %4 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
              %5 = quantum.insert %0[ 0], %out_qubits_22#0 : !quantum.reg, !quantum.bit
              %6 = quantum.insert %5[ 1], %out_qubits_22#1 : !quantum.reg, !quantum.bit
              %7 = quantum.insert %6[ 2], %out_qubits_22#2 : !quantum.reg, !quantum.bit
              %8 = quantum.insert %7[ 3], %out_qubits_22#3 : !quantum.reg, !quantum.bit
              %9 = quantum.extract %8[ 0] : !quantum.reg -> !quantum.bit
              %10 = quantum.namedobs %9[ PauliZ] : !quantum.obs
              %11 = quantum.expval %10 : f64
              %from_elements = tensor.from_elements %11 : tensor<f64>
              %12 = quantum.insert %8[ 0], %9 : !quantum.reg, !quantum.bit
              quantum.dealloc %12 : !quantum.reg
              quantum.device_release
              return %from_elements : tensor<f64>
            }
    )";

    auto parsed_ops = parse_quantum_custom_ops(mlir_code);

    // std::cout << "Parsed quantum.custom operations:" << std::endl;
    // for (const auto& op : parsed_ops) {
    //     std::cout << op.getName() << std::endl;
    // }

    // Define Operators
    Operator single_exc("SingleExcitation");
    Operator single_exc_plus("SingleExcitationPlus");
    Operator double_exc("DoubleExcitation");
    Operator cry("CRY");
    Operator s("S");
    Operator phase("PhaseShift");
    Operator rz("RZ");
    Operator rx("RX");
    Operator ry("RY");
    Operator rot("Rot");
    Operator hadamard("Hadamard");
    Operator cnot("CNOT");
    Operator cy("CY");
    Operator t("T");
    Operator global_phase("GlobalPhase");
    Operator phaseshift("PhaseShift");

    // ('SingleExcitation', {H:2, CNOT:2, RY:2}, _single_excitation_decomp)
    ResourceOp res_single_exc({{hadamard, 2}, {cnot, 2}, {ry, 2}});
    RuleRefOp rule_single_exc(single_exc, res_single_exc, "_single_excitation_decomp");

    // ('SingleExcitationPlus', {H:2, CY:1, CNOT:2, RY:2, S:1, RZ:1, GlobalPhase:1}, _single_excitation_plus_decomp)
    ResourceOp res_single_exc_plus({
        {hadamard, 2}, {cy, 1}, {cnot, 2}, {ry, 2},
        {s, 1}, {rz, 1}, {global_phase, 1}});
    RuleRefOp rule_single_exc_plus(single_exc_plus, res_single_exc_plus, "_single_excitation_plus_decomp");

    // ('DoubleExcitation', {CNOT:14, H:6, RY:8}, _doublexcit)
    ResourceOp res_double_exc1({{cnot, 14}, {hadamard, 6}, {ry, 8}});
    RuleRefOp rule_double_exc1(double_exc, res_double_exc1, "_doublexcit");

    // ('CRY', {RY:2, CNOT:2}, _cry)
    ResourceOp res_cry({{ry, 2}, {cnot, 2}});
    RuleRefOp rule_cry(cry, res_cry, "_cry");

    // ('S', {PhaseShift:1}, _s_phaseshift)
    ResourceOp res_s1({{phase, 1}});
    RuleRefOp rule_s1(s, res_s1, "_s_phaseshift");

    // ('S', {T:1}, _s_to_t)
    ResourceOp res_s2({{t, 1}});
    RuleRefOp rule_s2(s, res_s2, "_s_to_t");

    // ('PhaseShift', {RZ:1, GlobalPhase:1}, _phaseshift_to_rz_gp)
    ResourceOp res_phase({{rz, 1}, {global_phase, 1}});
    RuleRefOp rule_phase(phase, res_phase, "_phaseshift_to_rz_gp");

    // ('RZ', {Rot:1}, _rz_to_rot)
    ResourceOp res_rz1({{rot, 1}});
    RuleRefOp rule_rz1(rz, res_rz1, "_rz_to_rot");

    // ('RZ', {RY:2, RX:1}, _rz_to_ry_rx)
    ResourceOp res_rz2({{ry, 2}, {rx, 1}});
    RuleRefOp rule_rz2(rz, res_rz2, "_rz_to_ry_rx");

    // ('Rot', {RZ:2, RY:1}, _rot_to_rz_ry_rz)
    ResourceOp res_rot({{rz, 2}, {ry, 1}});
    RuleRefOp rule_rot(rot, res_rot, "_rot_to_rz_ry_rz");


    std::vector<Operator> ops(parsed_ops.begin(), parsed_ops.end());
    std::vector<Operator> gateset = {ry, rx, cnot, hadamard, global_phase};
    std::vector<RuleRefOp> rules = {
        rule_single_exc, rule_single_exc_plus,
        rule_double_exc1,
        rule_cry, rule_s1, rule_s2,
        rule_phase, rule_rz1, rule_rz2,
        rule_rot
    };

    Solver solver(ops, gateset, rules);
    auto solutions = solver.solve();
    // solver.show();

    assert(solutions.size() == 9);
    assert(solutions[single_exc] == "_single_excitation_decomp");
    assert(solutions[double_exc] == "_doublexcit");
    assert(solutions[rz] == "_rz_to_rot");
    assert(solutions[s] == "_s_phaseshift");
    assert(solutions[global_phase] == "base_op");
    assert(solutions[hadamard] == "base_op");
    assert(solutions[ry] == "base_op");
    assert(solutions[rx] == "base_op");
    assert(solutions[cnot] == "base_op");

    std::cout << "[PASS] Solver tests (4)" << std::endl;

}


int main() {
    test_operator();
    test_resourceop();
    test_rulerefop();
    test_solver1();
    test_solver2();
    test_solver3();
    test_solver4();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}

