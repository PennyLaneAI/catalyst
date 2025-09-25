
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>
#include <limits>
#include <queue>

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



// Solver
// _________________________________________
// * Ops<vector<Operator>>: Operators
// * Gateset<vector<string>>: Operators
// * Rules<vector<RuleRefOp>>: Rules
// + graph(): void
// + show(): stdout
// + solve(): map<Operator, RuleRef>

class Solver {
    std::vector<Operator> ops;
    std::vector<std::string> gateset;
    std::vector<RuleRefOp> rules;

    // Cached solutions map
    std::unordered_map<Operator, std::string> solutions;

public:
    Solver(const std::vector<Operator>& ops,
                const std::vector<std::string>& gateset,
                const std::vector<RuleRefOp>& rules)
        : ops(ops), gateset(gateset), rules(rules) {}


    std::unordered_map<Operator, std::string> solve() {
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
    std::vector<std::string> gateset = {"CNOT", "H"};
    std::vector<RuleRefOp> rules = {rule1, rule2};

    Solver solver(ops, gateset, rules);
    auto solutions = solver.solve();
    assert(solutions.size() == 2);
    assert(solutions[cz] == "cz_decomp_rule");
    assert(solutions[h] == "h_rule");
    // solver.show();

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

    ResourceOp h_to_rz_rx_rz({{rz, 2}, {rx, 1}});
    RuleRefOp rule3(h, h_to_rz_rx_rz, "h_rz_rx_rz_rule");

    ResourceOp h_to_rz_rz({{rz, 2}});
    RuleRefOp rule4(h, h_to_rz_rz, "h_rz_rz_rule");

    ResourceOp rz_self({{rz, 1}});
    RuleRefOp rule5(rz, rz_self, "rz_rule");

    ResourceOp rx_self({{rx, 1}});
    RuleRefOp rule6(rx, rx_self, "rx_rule");

    ResourceOp cnot_self({{cnot, 1}});
    RuleRefOp rule7(cnot, cnot_self, "cnot_rule");

    std::vector<Operator> ops = {cz, h};
    std::vector<std::string> gateset = {"CNOT", "RZ", "RX"};
    std::vector<RuleRefOp> rules = {rule1, rule2, rule3, rule4, rule5, rule6, rule7};

    Solver solver(ops, gateset, rules);

    auto solutions = solver.solve();
    assert(solutions.size() == 2);
    assert(solutions[cz] == "cz_h_cnot_rule");
    assert(solutions[h] == "h_rz_rz_rule");
    // solver.show();

    std::cout << "[PASS] Solver tests (2)" << std::endl;
}


int main() {
    test_operator();
    test_resourceop();
    test_rulerefop();
    test_solver1();
    test_solver2();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}

