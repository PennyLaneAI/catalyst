from dataclasses import dataclass
@dataclass
class traversalState:
    
    name: str 
    
    tree_depth: int
    
    # 0=unvisited, 1=left_visited, 2=both_visited, "for_loop"=special marker
    visited: list
    
    # 0=normal, "for_loop"=special marker
    special: list
    
    # 0=normal, state = new nested state
    state_to_special: list
    
    # probs for each
    probs: list
    
    # segment table name
    segment_table: str
    
    def __repr__(self):
        state_to_special_str = [s.name if s != 0 else s for s in self.state_to_special]
        return f"Name: {self.name}, tree_depth: {self.tree_depth},\n\t visited: {self.visited},\n\t special: {self.special},\n\t probs: {self.probs}, \n\t segment_table: {self.segment_table} \n\t state_to_special: {state_to_special_str}"

class traversalStateForLoop(traversalState):
    for_loop_visited: int = 0
    last_computed_depth: int = 1
    
    def __repr__(self):
        return f"{super().__repr__()}, \n\t for_loop_visited: {self.for_loop_visited}, last_computed_depth: {self.last_computed_depth}"


# The depth is mcm * iterations + 1
# The plus 1 is for the initial segment function before the mcms


whitespace = " " * 4

def indent_print(text, indent=0, real_print=False):
    if real_print:
        print(whitespace * indent + text)
    # print if text contins "*** MEASURE ***
    if "*** MEASURE ***" in text:
        print(whitespace * indent + text)
    # else:
    #     print(whitespace * indent + ".")
def status_print(tt_state, depth):
    return f"[{tt_state.name}]: visited = {tt_state.visited} at depth {depth}"

def segment_table(depth, segment_table, postselect=None):
    if depth == 0:
        # call segment table
        indent_print(f"... ops in table {segment_table}", indent=depth)
    else:
        # call segment table
        assert postselect is not None
        indent_print(f"... mcm postselect = {postselect}", indent=depth)
        indent_print(f"... ops in table [[{segment_table}]]", indent=depth)

def do_prob(probs,depth):
    # assume mcm_qbit is always 0

    # call probs function to get the prob
    # prob = qml.probs(mcm_qubit)
    # For demo, we use a fixed prob based on the depth
    prob = probs[depth]
    # indent_print(f"probs(0) = {prob}", indent=depth)

    if prob[0] == 1.0:
        postselect = 0
        visited_state = 2
        # indent_print("mcm always 0", indent=depth)
        return postselect, visited_state
    elif prob[1] == 1.0:
        postselect = 1
        visited_state = 2
        # indent_print("mcm always 1", indent=depth)
        return postselect, visited_state
    else:
        postselect = 0
        visited_state = 1
        return postselect, visited_state


def simulation(case, depth, tt_state):
    # indent_print(f"[before] | {status_print(tt_state, depth)}/{tt_state.tree_depth-1}", indent=depth)

    # only need to run the first segment function, no mcm
    if depth == 0:
        # generate call.segment_table(depth, ...)
        # and adjust visited state
        segment_table(depth, tt_state.segment_table)
        tt_state.visited[depth] = 2
        # indent_print(f"[update] | {status_print(tt_state, depth)}", indent=depth)
        return

    # otherwise
    if case == 0:
        # not visited before
        postselect, visited_state = do_prob(tt_state.probs, depth)

        # update visited state, it could be 1, or 2 just based on the prob
        # if it transfer to 2, then it mean it only have 1 branch to go, we mark it completed at
        # the point
        tt_state.visited[depth] = visited_state
        # need to store the state if it is 1, since it is expected to have right branch
        if visited_state == 1:
            store_state(depth)

        segment_table(depth, tt_state.segment_table, postselect=postselect)
    elif case == 1:
        tt_state.visited[depth] = 2 # go from 1 to 2 to mark it completed

        # case 1 always restore the state! Since if it don't need to restore the state,
        # then it should be 2 already, couldn't run into this case
        restore_state(depth)

        # we mark the postselect as 1, since it is always the right branch
        segment_table(depth, tt_state.segment_table, postselect=1)
    else:
        indent_print("error case")
        exit(1)

    # indent_print(f"[update] | {status_print(tt_state, depth)}", indent=depth)

def store_state(depth):
    indent_print(f"storing state at depth: {depth}", indent=depth)

def restore_state(depth):
    indent_print(f"restoring state at depth: {depth}", indent=depth)

def for_loop_simulation(global_depth, for_loop_state, tt_state):
    
    indent_print(f"F_SIM | {status_print(for_loop_state, global_depth)},", indent=global_depth, real_print=True)
    
    # if all elements are 2 in visited
    if all(v == 2 for v in for_loop_state.visited):
        
        # Reset for future traversals
        for_loop_state.last_computed_depth = 1  
        for_loop_state.visited = [0 for _ in for_loop_state.visited]


    tree_traversal_one_way(global_depth, for_loop_state)

    indent_print(f"F_SIM over | {status_print(for_loop_state, global_depth)}, return 1", indent=global_depth, real_print=True)

    return 1  # Mark the for_loop as left visited
    


def tree_traversal_one_way(global_depth, tt_state):
    
    if tt_state.visited[0] == 0:
        # Just run the first segment function, no mcm
        simulation(0, 0, tt_state)
    
    depth = tt_state.last_computed_depth

    special_state = None

    # Main loop
    while depth >= 0:
        indent_print(f"OW [before] |  {status_print(tt_state, depth) } | g_depth: {global_depth+depth}", indent=global_depth+depth, real_print=True)

        # Before region: check if hit leaf
        if depth == tt_state.tree_depth:
            break

        # Body region: process current node
        status = tt_state.visited[depth]
        depth_type = tt_state.special[depth]
        
        if depth_type == "for_loop" and status < 2:
            
            if tt_state.visited[depth] == 0:
                print("Start a new for loop traversal")
                tt_state.visited[depth] += 1            

            special_state = tt_state.state_to_special[depth]
            
            special_status = for_loop_simulation(global_depth+depth, special_state, tt_state)
            
            if all(v == 2 for v in special_state.visited):
                tt_state.visited[depth] = 2
                            
            if special_status < 2:
                tt_state.last_computed_depth = depth
                depth += 1

            if special_status == 2:
                print("Update a new for loop traversal")
                tt_state.visited[depth] = 2
                
            indent_print(f"OW [after loop] |  {status_print(tt_state, depth)},special status: {special_status}", indent=global_depth+depth, real_print=True)
            continue

        if status < 2:  # Case 0: unvisited or left visited
            simulation(status, depth, tt_state)
            tt_state.last_computed_depth = depth
            depth += 1
        
        elif status == 2:  # Case 2: both visited -> backtrack
            tt_state.visited[depth] = 0
            depth = depth - 1

        else:  # Error case
            depth = -1
            
        indent_print(f"OW [after] |  {status_print(tt_state, depth)},", indent=global_depth+depth, real_print=True)


def tree_traversal(tt_state):
    depth = 1

    # Just run the first segment function, no mcm
    simulation(0, 0, tt_state)

    # Main loop
    while depth >= 0:
        
        # Before region: check if hit leaf
        # Measure point
        if depth == tt_state.tree_depth:
            print("#"*100)            
            print(f"{main_state.name} |  {for_loop_state.name} | {for_loop_state_nested.name}")
            print(f"{main_state.visited}  | {for_loop_state.visited} | {for_loop_state_nested.visited}")

            indent_print(f"*** MEASURE *** {depth}", indent=depth)
            print("#"*100)
            depth = depth - 1

        indent_print(f"TT [before] |  {status_print(tt_state, depth)},", indent=depth, real_print=True)

        # Body region: process current node
        status = tt_state.visited[depth]
        depth_type = tt_state.special[depth]
        
        if depth_type == "for_loop" and status < 2:
            
            if tt_state.visited[depth] == 0:
                print("Start a new for loop traversal")
                tt_state.visited[depth] += 1            

            special_state = tt_state.state_to_special[depth]

            special_status = for_loop_simulation(depth, special_state, tt_state)
     
            if all(v == 2 for v in special_state.visited):
                tt_state.visited[depth] = 2
     
            if special_status < 2:
                depth += 1
            elif special_status == 2:
                tt_state.visited[depth] = 2
                
            indent_print(f"TT [iter loop] |  {status_print(tt_state, depth)}, special status: {special_status}", indent=depth, real_print=True)
            continue

        if status < 2:  # Case 0: unvisited or left visited
            simulation(status, depth, tt_state)
            depth += 1

        elif status == 2:  # Case 2: both visited -> backtrack
            tt_state.visited[depth] = 0
            depth -= 1

        else:  # Error case
            depth = -1


        indent_print(f"TT [iter] |  {status_print(tt_state, depth)},", indent=depth, real_print=True)


# if True:
if __name__ == "__main__":
    # Nested for loop state
    for_loop_mcm = 2
    for_loop_iterations = 2

    for_loop_state_nested = traversalStateForLoop(
        name="for_loop_state_nested",
        tree_depth = for_loop_mcm*for_loop_iterations + 1,
        visited = [0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],

        # Repeat the special for each iteration
        special=[ 0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        state_to_special=[ 0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        probs=[[0.5, 0.5] for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        segment_table="for_loop_segment_table_nested"
    )    


    # First for loop state
    for_loop_mcm = 2
    for_loop_iterations = 2

    for_loop_state = traversalStateForLoop(
        name="for_loop_state_top",
        tree_depth = for_loop_mcm*for_loop_iterations + 1,
        visited = [0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],

        # Repeat the special for each iteration
        special=[ 0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        state_to_special=[ 0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        probs=[[0.5, 0.5] for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        segment_table="for_loop_segment_table"
    )    

    for_loop_state.special[-2] = "for_loop"
    for_loop_state.state_to_special[-2] = for_loop_state_nested

    # Main traversal state
    main_mcm = 2
    main_state = traversalState(
        name="main_state",
        tree_depth= main_mcm + 1,
        visited=[0 for _ in range(main_mcm + 1)],
        special=[0 for i in range(main_mcm + 1)],
        state_to_special=[0 for i in range(main_mcm + 1)],
        # special=[0 for _ in range(main_mcm + 1)],
        probs=[[0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6]],
        segment_table="main_segment_table"
    )    

    main_state.special[-2] = "for_loop"
    main_state.state_to_special[-2] = for_loop_state

    



# if __name__ == "__main__":

    print(main_state)
    print(for_loop_state)
    print(for_loop_state_nested)
    print("-"*100)
    print("-"*100)
    # exit(0)

    tree_traversal(main_state)
