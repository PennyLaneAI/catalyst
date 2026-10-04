visited = [0, 0, 0, 0]  # 0=unvisited, 1=left_visited, 2=both_visited
tree_depth = 4

probs = [[1000, -1000], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9], [0.4, 0.6]]
exp_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # The expected values at the leaves
folded_result = [0] * (tree_depth + 1)
saved_probs = [1] * (tree_depth + 1)

index = [0]
whitespace = " " * 4


def indent_print(text, indent=0):
    print(whitespace * indent + text)


# the main logic for state transition
def traversal_handling_func(qubit, depth, branch):
    def scfIf(qubit, depth, branch):
        if branch == 0:
            # assume probs on qubit
            prob = probs[depth]
            indent_print(f"probs({qubit}) = {prob}", indent=depth)
            if prob[1] == 0.0:
                indent_print("mcm always 0", indent=depth)
                indent_print(f"save probs[{depth}] = prob[0]", indent=depth)
                saved_probs[depth + 1] = prob[0]
                return 0, 2
            elif prob[0] == 0.0:
                indent_print("mcm always 1", indent=depth)
                indent_print(f"save probs[{depth}] = prob[1]", indent=depth)
                saved_probs[depth + 1] = prob[1]
                return 1, 2
            else:
                store_state(depth)
                indent_print(f"save probs[{depth}] = prob[0]", indent=depth)
                saved_probs[depth + 1] = prob[0]
                return 0, 1
        else:
            # TODO: probs
            saved_probs[depth + 1] = 1.0 - saved_probs[depth + 1]

            restore_state(depth)
            return 1, 2

    postelect, visited_state = scfIf(qubit, depth, branch)
    visited[depth] = visited_state
    indent_print(f"[update] visited: {visited}, depth: {depth}", indent=depth)
    return postelect


def segment_0(depth, branch):
    visited[depth] = 2
    indent_print(f"... ops", indent=depth)


def segment_1(depth, branch):
    qubit = 0
    postselect = traversal_handling_func(qubit, depth, branch)
    indent_print(f"... mcm postselect = {postselect}", indent=depth)
    indent_print(f"... ops", indent=depth)


def segment_2(depth, branch):
    qubit = 0
    postselect = traversal_handling_func(qubit, depth, branch)
    indent_print(f"... mcm postselect = {postselect}", indent=depth)
    indent_print(f"... ops", indent=depth)


def segment_3(depth, branch):
    qubit = 0
    postselect = traversal_handling_func(qubit, depth, branch)

    indent_print(f"... mcm postselect = {postselect}", indent=depth)
    indent_print(f"... ops", indent=depth)
    indent_print(f"... expval", indent=depth)

    # 1. extract the f64 from tensor<f64>
    # 2. store the f64 to the folded_result
    exp_val = exp_vals[index[0]]
    index[0] += 1
    folded_result[depth + 1] = folded_result[depth + 1] + exp_val * saved_probs[depth + 1]
    indent_print(f"update folded_result[{depth + 1}]: {folded_result[depth + 1]}", indent=depth)


def segment_table(depth, branch):
    reg = 0
    exp = 0.5

    match depth:
        case 0:
            segment_0(depth, branch)
        case 1:
            segment_1(depth, branch)
        case 2:
            segment_2(depth, branch)
        case 3:
            segment_3(depth, branch)

    return reg, exp


def simulation(case, depth):
    indent_print(f"[before] visited: {visited}, depth: {depth}", indent=depth)

    # Case 0: Unvisited node
    if case < 2:
        reg, exp = segment_table(depth, case)
        depth += 1
        return depth

    # Case 2: Finished - goes up
    elif case == 2:
        # Reset visited status
        visited[depth] = 0
        # Move up one level
        indent_print(f"[update] Goes up, visited: {visited}, depth: {depth}", indent=depth)

        # TODO: calculate folded_result
        folded_result[depth] += saved_probs[depth] * folded_result[depth + 1]
        folded_result[depth + 1] = 0
        indent_print(f"go up, folded_result[{depth}]: {folded_result[depth]}", indent=depth)

        depth -= 1

        return depth
    else:
        depth = -1
        assert False, "error case"


def store_state(depth):
    indent_print(f"storing state at depth: {depth}", indent=depth)


def restore_state(depth):
    indent_print(f"restoring state at depth: {depth}", indent=depth)


def tree_traversal():
    depth = 0

    # Main loop
    while True:
        # before_region: check if hit leaf
        if depth == tree_depth:
            depth = depth - 1

        if depth < 0:
            break

        status = visited[depth]
        depth = simulation(status, depth)

    print(f"Final result: {folded_result[0]}")


if __name__ == "__main__":
    tree_traversal()
