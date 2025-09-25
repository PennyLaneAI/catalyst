
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>


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

    const Operator getOperator() const { return op; }
    const ResourceOp getResources() const { return resources; }
    const std::string getRuleRef() const { return rule_ref; }
};



// Solver
// _________________________________________
// * Ops<vector<Operator>>: Operators
// * Gateset<vector<string>>: Operators
// * Rules<vector<RuleRefOp>>: Rules
// + graph(): void
// + show(): stdout
// + solve(): map<Operator, RuleRefOp>

class Solver {
    std::vector<Operator> ops;
    std::vector<std::string> gateset;
    std::vector<RuleRefOp> rules;

public:
    Solver(const std::vector<Operator>& ops,
                const std::vector<std::string>& gateset,
                const std::vector<RuleRefOp>& rules)
        : ops(ops), gateset(gateset), rules(rules) {}

    void graph() {
        // Placeholder for graph generation logic
        std::cout << "Graph generation not implemented.\n";
    }

    std::unordered_map<Operator, std::string> solve() {
        std::unordered_map<Operator, std::string> solutions;

        for (const auto& rule: rules) {
            solutions[rule.getOperator()] = rule.getRuleRef();
        }
        return solutions;
    }

    void show() {
        std::cout << "Not implemented.\n";
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

void test_solver() {
    Operator op1("H"), op2("X");
    ResourceOp r1({{op1, 1}});
    ResourceOp r2({{op2, 2}});

    RuleRefOp rr1(op1, r1, "ruleH");
    RuleRefOp rr2(op2, r2, "ruleX");

    Solver solver({op1, op2}, {"H", "X"}, {rr1, rr2});
    // solver.graph();
    // solver.show();

    auto solution = solver.solve();
    assert(solution.at(op1) == "ruleH");
    assert(solution.at(op2) == "ruleX");

    std::cout << "[PASS] Solver tests" << std::endl;
}

int main() {
    test_operator();
    test_resourceop();
    test_rulerefop();
    test_solver();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}

