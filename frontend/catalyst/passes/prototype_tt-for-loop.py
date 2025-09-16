visited = [0, 0, 0, 0]  # 0=unvisited, 1=left_visited, 2=both_visited
tree_depth = 4

# probs for all qbit 0
probs = [[0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6]]


whitespace = " " * 4

def indent_print(text, indent=0):
    print(whitespace * indent + text)

def segment_table(depth, postselect=None):
    if depth == 0:
        # call segment table
        indent_print("... ops", indent=depth)
    else:
        # call segment table
        assert postselect is not None
        indent_print(f"... mcm postselect = {postselect}", indent=depth)
        indent_print("... ops", indent=depth)

def do_prob(depth):
    # assume mcm_qbit is always 0

    # call probs function to get the prob
    # prob = qml.probs(mcm_qubit)
    # For demo, we use a fixed prob based on the depth
    prob = probs[depth]
    indent_print(f"probs(0) = {prob}", indent=depth)

    if prob[0] == 1.0:
        postselect = 0
        visited_state = 2
        indent_print("mcm always 0", indent=depth)
        return postselect, visited_state
    elif prob[1] == 1.0:
        postselect = 1
        visited_state = 2
        indent_print("mcm always 1", indent=depth)
        return postselect, visited_state
    else:
        postselect = 0
        visited_state = 1
        return postselect, visited_state


def simulation(case, depth):
    indent_print(f"[before] visited: {visited}, depth: {depth}", indent=depth)

    # only need to run the first segment function, no mcm
    if depth == 0:
        # generate call.segment_table(depth, ...)
        # and adjust visited state
        segment_table(depth)
        visited[depth] = 2
        indent_print(f"[update] visited: {visited}, depth: {depth}", indent=depth)
        return

    # otherwise
    if case == 0:
        # not visited before
        postselect, visited_state = do_prob(depth)

        # update visited state, it could be 1, or 2 just based on the prob
        # if it transfer to 2, then it mean it only have 1 branch to go, we mark it completed at
        # the point
        visited[depth] = visited_state
        indent_print(f"[update] visited: {visited}, depth: {depth}", indent=depth)
        # need to store the state if it is 1, since it is expected to have right branch
        if visited_state == 1:
            store_state(depth)

        segment_table(depth, postselect=postselect)
    elif case == 1:
        visited[depth] = 2 # go from 1 to 2 to mark it completed
        indent_print(f"[update] visited: {visited}, depth: {depth}", indent=depth)

        # case 1 always restore the state! Since if it don't need to restore the state,
        # then it should be 2 already, couldn't run into this case
        restore_state(depth)

        # we mark the postselect as 1, since it is always the right branch
        segment_table(depth, postselect=1)
    else:
        indent_print("error case")
        exit(1)

def store_state(depth):
    indent_print(f"storing state at depth: {depth}", indent=depth)

def restore_state(depth):
    indent_print(f"restoring state at depth: {depth}", indent=depth)

def tree_traversal():
    depth = 1

    # Just run the first segment function, no mcm
    simulation(0, 0)

    # Main loop
    while depth >= 0:
        # Before region: check if hit leaf
        if depth == tree_depth:
            depth = depth - 1

        # Body region: process current node
        status = visited[depth]

        if status < 2:  # Case 0: unvisited or left visited
            simulation(status, depth)
            depth += 1

        elif status == 2:  # Case 2: both visited -> backtrack
            visited[depth] = 0
            depth = depth - 1

        else:  # Error case
            depth = -1


if __name__ == "__main__":
    tree_traversal()
