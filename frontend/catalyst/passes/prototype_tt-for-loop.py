from dataclasses import dataclass, field
@dataclass
class traversalState:
    
    name: str 
    # 0=unvisited, 1=left_visited, 2=both_visited, "for_loop"=special marker
    visited: list = field(default_factory=lambda: [0])
    # 0=normal, "for_loop"=special marker
    special: list = field(default_factory=lambda: [0])
    # 0=normal, state = new nested state
    state_to_special: list = field(default_factory=lambda: [0])
    # probs for each
    probs: list = field(default_factory=lambda: [[1.0,0.0]])
    # segment table name
    segment_table: list = field(default_factory=lambda: ["seg_A"])
    tree_depth: int = 1
    last_computed_depth: int = 0
    tree_section: str = "main"
    tree_repetition: int = 1
    
    def __repr__(self):
        state_to_special_str = [s.name if s != 0 else s for s in self.state_to_special]
        output = f"Name: {self.name}, tree_depth: {self.tree_depth}\n\t"
        output += f"tree_section: {self.tree_section}\n\t"
        output += f"tree_repetition: {self.tree_repetition}\n\t"
        output += f"visited: {self.visited},\n\t"
        output += f"special: {self.special},\n\t"
        output += f"probs: {self.probs}, \n\t"
        output += f"segment_table: {self.segment_table} \n\t"
        output += f"state_to_special: {state_to_special_str} \n\t"
        output += f"last_computed_depth: {self.last_computed_depth}"
        return output


whitespace = " " * 4

def indent_print(text, indent=0, real_print=False):
    if real_print:
        print(whitespace * indent + text)
    if "*** MEASURE ***" in text:
        print(whitespace * indent + text)

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

def do_prob(probs,depth, mcm_qbit):
    # assume mcm_qbit is always 0

    # call probs function to get the prob
    # prob = qml.probs(mcm_qubit)
    # For demo, we use a fixed prob based on the depth
    prob = probs[depth]
    indent_print(f"probs({mcm_qbit}) = {prob}", indent=depth)

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


    # otherwise
    if case == 0:
        # not visited before
    # only need to run the first segment function, no mcm
        if depth == 0:
            # generate call.segment_table(depth, ...)
            # and adjust visited state
            segment_table(depth, tt_state.segment_table)
            tt_state.visited[depth] = 2
            # indent_print(f"[update] | {status_print(tt_state, depth)}", indent=depth)
            return depth + 1
        else:
            mcm_qbit = 0
            postselect, visited_state = do_prob(tt_state.probs, depth, mcm_qbit)

            # update visited state, it could be 1, or 2 just based on the prob
            # if it transfer to 2, then it mean it only have 1 branch to go, we mark it completed at
            # the point
            tt_state.visited[depth] = visited_state
            # need to store the state if it is 1, since it is expected to have right branch
            if visited_state == 1:
                store_state(depth)

            segment_table(depth, tt_state.segment_table, postselect=postselect)
            depth += 1
        return depth

    elif case == 1:
        tt_state.visited[depth] = 2 # go from 1 to 2 to mark it completed

        # case 1 always restore the state! Since if it don't need to restore the state,
        # then it should be 2 already, couldn't run into this case
        restore_state(depth)

        # we mark the postselect as 1, since it is always the right branch
        segment_table(depth, tt_state.segment_table, postselect=1)
        depth += 1
        return depth
    elif case == 2:
        tt_state.visited[depth] = 0
        depth -= 1
        return depth

    else:
        assert False, "error case"

    # indent_print(f"[update] | {status_print(tt_state, depth)}", indent=depth)

def store_state(depth):
    indent_print(f"storing state at depth: {depth}", indent=depth)

def restore_state(depth):
    indent_print(f"restoring state at depth: {depth}", indent=depth)

debug_stop = [0]

def print_visited_state(tt_state):
    print(f"Visited state of {tt_state.name}: {tt_state.visited}")
    print(f"Special state of {tt_state.name}: {tt_state.special}")
    for i in tt_state.state_to_special:
        if i != 0:
            print_visited_state(i)

def tree_traversal(tt_state, global_depth=0, one_way=False):

    depth = tt_state.last_computed_depth

    # print(f"FDX: Start traversal of {tt_state.name} from depth {depth} with visited {tt_state.visited}")
    # Main loop
    while depth >= 0:
        
        # Before region: check if hit leaf
        # Measure point
        if depth == tt_state.tree_depth:
            
            if one_way:
                tt_state.last_computed_depth = depth - 1 
                break
            else:            
                print("#"*100)            
                print_visited_state(tt_state)
                # print(f"{main_state.name} |  {for_loop_state_fdx.name} | {for_loop_state_fdx_nested.name}")
                # print(f"{main_state.visited}  | {for_loop_state_fdx.visited} | {for_loop_state_fdx_nested.visited}")

                indent_print(f"*** MEASURE *** {depth}", indent=depth)
                print("#"*100)
                depth = depth - 1

        indent_print(f"TT [before] |  {status_print(tt_state, depth) } | g_depth: {global_depth+depth}", indent=global_depth+depth, real_print=True)

        # Body region: process current node
        status = tt_state.visited[depth]
        depth_type = tt_state.special[depth]
        
        if depth_type == "for_loop" and status < 2:  # Special for loop handling
            
            # First time visit the for loop node
            if tt_state.visited[depth] == 0:
                tt_state.visited[depth] = 1            

            # Get the special TT state
            special_state = tt_state.state_to_special[depth]

            # If the special state has completed a full traversal, reset it
            if all(v == 2 for v in special_state.visited):
                # Reset for future traversals
                special_state.last_computed_depth = 0
                special_state.visited = [0 for _ in special_state.visited]

            # Run the special state traversal
            tree_traversal(special_state, global_depth=depth, one_way=True)

            # After return from the special state, check if it has fully visited
            if all(v == 2 for v in special_state.visited):
                tt_state.visited[depth] = 2

            # Update the last computed depth
            tt_state.last_computed_depth = depth
            depth += 1
                
            indent_print(f"TT [after loop] |  {status_print(tt_state, depth)}", indent=global_depth+depth, real_print=True)
            
            continue

        depth = simulation(status, depth, tt_state)
        tt_state.last_computed_depth = depth

        indent_print(f"TT [iter] |  {status_print(tt_state, depth)},", indent=global_depth+depth, real_print=True)


def test_for_loop_traversal():
    # Nested_2 for loop state
    for_loop_mcm = 2
    for_loop_iterations = 2

    for_loop_state_fdx_nested_02 = traversalState(
        name="for_loop_state_nested_02",
        tree_depth = for_loop_mcm*for_loop_iterations + 1,
        visited = [0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],

        # Repeat the special for each iteration
        special=[ 0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        state_to_special=[ 0 for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        probs=[[0.5, 0.5] for _ in range(for_loop_mcm*for_loop_iterations + 1)],
        segment_table="for_loop_segment_table_nested"
    )    

    
    # Nested for loop state
    for_loop_mcm = 2
    for_loop_iterations = 2
    name="for_loop_state_nested"
    
    for_loop_state_fdx_nested = traversalState(
        name=name,
        tree_depth = for_loop_mcm + 1,
        visited = [0 for _ in range(for_loop_mcm + 1)],

        # Repeat the special for each iteration
        special=[ 0 for _ in range(for_loop_mcm + 1)],
        state_to_special=[ 0 for _ in range(for_loop_mcm + 1)],
        probs=[[0.5, 0.5] for _ in range(for_loop_mcm + 1)],
        segment_table=[name+"_"+chr(ord('A') + i) for i in range(for_loop_mcm + 1)],
        tree_section = "for_loop",
        tree_repetition = for_loop_iterations
    )    
    for_loop_state_fdx_nested.probs[0] = [1.0, 0.0]  # always go left first

    # for_loop_state_fdx_nested.special[-1] = "for_loop"
    # for_loop_state_fdx_nested.state_to_special[-1] = for_loop_state_fdx_nested_02


    # First for loop state
    for_loop_mcm = 1
    for_loop_iterations = 3
    name = "for_loop_top"

    for_loop_state_fdx = traversalState(
        name=name,
        tree_depth = for_loop_mcm + 1,
        visited = [0 for _ in range(for_loop_mcm + 1)],

        # Repeat the special for each iteration
        special=[ 0 for _ in range(for_loop_mcm + 1)],
        state_to_special=[ 0 for _ in range(for_loop_mcm + 1)],
        probs=[[0.5, 0.5] for _ in range(for_loop_mcm + 1)],
        segment_table=[name+"_"+chr(ord('A') + i) for i in range(for_loop_mcm + 1)],
        tree_section = "for_loop",
        tree_repetition = for_loop_iterations
    )    

    for_loop_state_fdx.probs[0] = [1.0, 0.0]  # always go left first
    # for_loop_state_fdx.special[-1] = "for_loop"
    # for_loop_state_fdx.state_to_special[-1] = for_loop_state_fdx_nested

    # Main traversal state
    main_mcm = 1
    main_mcm += 1 # Add one for the for statement
    main_state = traversalState(
        name="main_state",
        tree_depth= main_mcm + 1,
        visited=[0 for _ in range(main_mcm + 1)],
        special=[0 for i in range(main_mcm + 1)],
        state_to_special=[0 for i in range(main_mcm + 1)],
        # special=[0 for _ in range(main_mcm + 1)],
        probs=[[0.5, 0.5] for _ in range(main_mcm + 1)],
        segment_table=["main_" + chr(ord('A') + i) for i in range(main_mcm + 1)]
    )    

    main_state.probs[0] = [1.0, 0.0]

    for_loop_position = -2
    main_state.special[for_loop_position] = "for_loop"
    main_state.state_to_special[for_loop_position] = for_loop_state_fdx
    for deep in range(main_state.tree_depth):
        if main_state.special[deep] == "for_loop":
            main_state.segment_table[deep] = main_state.state_to_special[deep].name
            


    print(main_state)
    print("-"*100)
    
    def expand_tree_for_loop(state):
        for i in range(state.tree_depth):
            if state.special[i] == "for_loop":
                state.tree_depth += 2
                state.visited.insert(i, 0)
                state.visited.insert(i+2, 0)
                state.special.insert(i, 0)
                state.special.insert(i+2, 0)
                state.probs.insert(i, [0.5, 0.5])
                state.probs.insert(i+2, [1.0, 0.0])
                state.segment_table.insert(i, state.state_to_special[i].name+"_expanded_A")
                state.segment_table.insert(i+2, state.state_to_special[i].name+"_expanded_B")

    # expand_tree_for_loop(main_state)

    
    print(main_state)
    print("-"*100)
    print(for_loop_state_fdx)
    print("-"*100)
    print(for_loop_state_fdx_nested)
    print("-"*100)
    # print(for_loop_state_fdx_nested_02)
    # print("-"*100)
                
    global_visited = []
    global_special = []
    global_probs = []
    global_segment_table = []
    global_state_to_special = []

    def add_tree_to_global(state):
        
        for i in range(state.tree_repetition):
            for i in range(state.tree_depth):                
                if state.special[i] == "for_loop":
                    add_tree_to_global(state.state_to_special[i])
                    continue
                
                global_visited.append(state.visited[i])
                global_special.append(state.special[i])
                global_probs.append(state.probs[i])
                global_segment_table.append(state.segment_table[i])

    add_tree_to_global(main_state)

    assert len(global_visited) == len(global_special) == len(global_probs) == len(global_segment_table)
    
    for i in range(len(global_visited)):
        print(f"Depth {i:>4}: visited={global_visited[i]}, special={global_special[i]}, probs={global_probs[i]}, segment_table={global_segment_table[i]:>30}")

def push_mcm_2_structure(state, mcm_count, position=None, op_type="mcm", new_state=None):
    
    position = position if position is not None else state.tree_depth
    
    if op_type == "for_loop":
        assert mcm_count == 1, "for_loop only support 1 mcm for now"
        assert new_state is not None, "for_loop need a new_state"
        
    if op_type == "for_loop":
        # Adding the for loop structure
        state.tree_depth += 1
        state.visited.insert(position, 0)
        state.special.insert(position, "for_loop")
        state.probs.insert(position, [1.0, 0.0])  # always go left first
        state.segment_table.insert(position, "for_loop")
        state.state_to_special.insert(position, new_state)


        state.tree_depth += 1
        state.visited.insert(position+1, 0)
        state.special.insert(position+1, 0)
        state.probs.insert(position+1, [1.0, 0.0])  # always go left first
        state.segment_table.insert(position+1, state.name + "_added_mcm")
        state.state_to_special.insert(position+1, 0)

    if op_type == "mcm":

        state.tree_depth += mcm_count
        for _ in range(mcm_count):
            state.visited.insert(position, 0)
            state.special.insert(position, 0)
            state.probs.insert(position, [0.5, 0.5])
            state.segment_table.insert(position, state.name + "_added_mcm")
            state.state_to_special.insert(position, 0)
        
    # rename segment tables
    for i in range(state.tree_depth):
        if state.special[i] != "for_loop":
            state.segment_table[i] = state.name + "_" + chr(ord('A') + i)

def expand_repetition(state):

    for i in state.state_to_special:
        if i != 0:
            expand_repetition(i)

    repetitions = state.tree_repetition
    
    state.tree_repetition = 1
    state.tree_depth = state.tree_depth * repetitions     
    
    state.visited = state.visited * repetitions
    state.special = state.special * repetitions
    state.probs = state.probs * repetitions
    state.segment_table = state.segment_table * repetitions
    state.state_to_special = state.state_to_special * repetitions
    


if __name__ == "__main__":

    print("-"*100)


    
    # Circuit Example
    
    #    Segment main_A
    #    mcm
    #    Segment main_B
    #   
    #    for loop start
    #      Segment FL_1_A
    #      mcm
    #      Segment FL_1_B
    #      mcm
    #      Segment FL_1_C
    #
    #    Segment main_D
    #    Segment main_E
    #    Segment main_F
    

    # Initialize the TraversalState | Segment A
    main_state = traversalState(
        name="main",
    )
    # Adding a mcm to main | Segment B
    push_mcm_2_structure(main_state, mcm_count=1)


    # Adding a for loop to main | for loop start | Segment FL_1_A and Segment main_D
    for_state_1 = traversalState(
        name="FL_1",
        tree_section="for_loop",
    )
    # Define iterations in the for loop state
    for_state_1.tree_repetition = 3
    push_mcm_2_structure(main_state, mcm_count=1, op_type="for_loop", new_state=for_state_1)
    # Adding mcm to for loop state | Segment FL_1_B
    push_mcm_2_structure(for_state_1, mcm_count=1)
    # Adding mcm to for loop state | Segment FL_1_C
    push_mcm_2_structure(for_state_1, mcm_count=1)
        
    # Adding another mcm to main
    push_mcm_2_structure(main_state, mcm_count=1) # Segment main_E
    push_mcm_2_structure(main_state, mcm_count=1) # Segment main_F

    # Final state
    print(main_state)
    print("-"*100)
    
    print(for_state_1)
    print("-"*100)


    global_visited = []
    global_special = []
    global_probs = []
    global_segment_table = []
    global_state_to_special = []

    def add_tree_to_global(state):
        
        for i in range(state.tree_repetition):
            for i in range(state.tree_depth):                
                if state.special[i] == "for_loop":
                    add_tree_to_global(state.state_to_special[i])
                    continue
                
                global_visited.append(state.visited[i])
                global_special.append(state.special[i])
                global_probs.append(state.probs[i])
                global_segment_table.append(state.segment_table[i])

    add_tree_to_global(main_state)

    assert len(global_visited) == len(global_special) == len(global_probs) == len(global_segment_table)
    

    global_state = traversalState(
        name="global",
        tree_depth=len(global_visited),
        visited=global_visited,
        special=global_special,
        state_to_special=[0 for _ in range(len(global_visited))],
        probs=global_probs,
        segment_table=global_segment_table
    )
    
    print(global_state)
    print("-"*100)

    for i in range(len(global_visited)):
        print(f"Depth {i:>4}: visited={global_visited[i]}, special={global_special[i]}, probs={global_probs[i]}, segment_table={global_segment_table[i]:>30}")



    # exit(0)

    # Case 1 : using single traversal though global state
    # tree_traversal(global_state)
    
    # Case 2 : using nested traversal
    expand_repetition(main_state)
    print(main_state)
    print("-"*100)
    expand_repetition(for_state_1)
    print(for_state_1)
    print("-"*100)
    
    # tree_traversal(main_state)
