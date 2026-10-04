from dataclasses import dataclass, field
import copy

# --------------------------------------------------------------------------------------------------
# Tree Traversal Printing
# --------------------------------------------------------------------------------------------------

whitespace = " " * 4

def indent_print(text, indent=0, real_print=False):
    if real_print:
        print(whitespace * indent + text)
    if "*** MEASURE ***" in text:
        print(whitespace * indent + text)

def status_print(tt_state, depth):
    return f"[{tt_state.name}]: visited = {tt_state.visited} at depth {depth}"

def print_visited_state(tt_state):
    print(f"Special state of {tt_state.name}: {tt_state.special}")
    print(f"Visited state of {tt_state.name}: {tt_state.visited}")
    for i in tt_state.state_to_special:
        if i != 0:
            print_visited_state(i)
            
def print_state(state):
    print(state)
    print("~"*100)
    for s in state.state_to_special:
        if s != 0:
            print_state(s)
            
# --------------------------------------------------------------------------------------------------
# Tree Traversal Logic
# --------------------------------------------------------------------------------------------------

# the main logic for state transition
def traversal_handling_func(qubit, depth, branch, tt_state):
    def scfIf(qubit, depth, branch):
        if branch == 0:
            # assume probs on qubit
            prob = tt_state.probs[depth]
            indent_print(f"probs({qubit}) = {prob}", indent=depth)
            if prob[1] == 0.0:
                indent_print("mcm always 0", indent=depth)
                indent_print(f"save probs[{depth}] = prob[0]", indent=depth)
                # saved_probs[depth + 1] = prob[0]
                return 0, 2
            elif prob[0] == 0.0:
                indent_print("mcm always 1", indent=depth)
                indent_print(f"save probs[{depth}] = prob[1]", indent=depth)
                # saved_probs[depth + 1] = prob[1]
                return 1, 2
            else:
                store_state(depth)
                indent_print(f"save probs[{depth}] = prob[0]", indent=depth)
                # saved_probs[depth + 1] = prob[0]
                return 0, 1
        else:
            # TODO: probs
            # saved_probs[depth + 1] = 1.0 - saved_probs[depth + 1]

            restore_state(depth)
            return 1, 2

    postelect, visited_state = scfIf(qubit, depth, branch)
    tt_state.visited[depth] = visited_state
    indent_print(f"[update] visited: {tt_state.visited}, depth: {depth}", indent=depth)
    return postelect

def select_segment(segment_name, segment_list):
    # substract hash_ from the begining of the segment name
    segment_name_real = segment_name.replace("hash_", "", 1)
    assert segment_name_real in segment_list, f"Segment {segment_name_real} not found in segment list"
    return segment_name_real

def segment_compute(depth, branch, segment_list, tt_state):
    if depth == 0 :
        tt_state.visited[depth] = 2
        segment_name = select_segment(tt_state.segment_table[depth], segment_list)
        indent_print(f"... ops {segment_name} at depth {depth}", indent=depth)
    else:
        qubit = 0
        postselect = traversal_handling_func(qubit, depth, branch, tt_state)
        indent_print(f"... mcm postselect = {postselect} at depth {depth}", indent=depth)
        segment_name = select_segment(tt_state.segment_table[depth], segment_list)
        indent_print(f"... ops {segment_name} at depth {depth}", indent=depth)

def segment_table(depth, branch, segment_list, tt_state):
    reg = 0 
    exp = 0.5
    
    if depth == 0:
        # call segment table
        segment_compute(0, branch, segment_list, tt_state)
    else:
        segment_compute(depth, branch, segment_list, tt_state)


def  simulation(case, depth, segment_list, tt_state):
    if case < 2:
        segment_table(depth, case, segment_list, tt_state)
        return depth + 1
    elif case == 2:
        tt_state.visited[depth] = 0
        return depth - 1
    else:
        depth = -1
        assert False, "error case"

def store_state(depth):
    indent_print(f"storing state at depth: {depth}", indent=depth)

def restore_state(depth):
    indent_print(f"restoring state at depth: {depth}", indent=depth)

def tree_traversal(tt_state, segment_list, global_depth=0, one_way=False):

    depth = tt_state.last_computed_depth

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
            tree_traversal(special_state, segment_list, global_depth=depth, one_way=True)

            # After return from the special state, check if it has fully visited
            if all(v == 2 for v in special_state.visited):
                tt_state.visited[depth] = 2

            # Update the last computed depth
            tt_state.last_computed_depth = depth
            depth += 1
                
            indent_print(f"TT [after loop] |  {status_print(tt_state, depth)}", indent=global_depth+depth, real_print=True)
            
            continue

        depth = simulation(status, depth, segment_list, tt_state)
        tt_state.last_computed_depth = depth

        indent_print(f"TT [iter] |  {status_print(tt_state, depth)},", indent=global_depth+depth, real_print=True)

# --------------------------------------------------------------------------------------------------
# Tree Traversal state manipulation
# --------------------------------------------------------------------------------------------------

@dataclass
class traversalState:
    
    # Name of the state
    name: str 

    # Depth of the tree
    tree_depth: int = 1

    # Visited status of each node
    # 0 = unvisited, 1 = left_visited, 2 = both_visited, "for_loop" = special marker
    visited: list = field(default_factory=lambda: [0])

    # Special status of each node
    # 0 = normal, "for_loop" = special marker
    special: list = field(default_factory=lambda: [0])

    # Link to special TT state
    # 0 = normal, state = new nested state
    state_to_special: list = field(default_factory=lambda: [0])
    
    # probs for each
    probs: list = field(default_factory=lambda: [[1.0,0.0]])
    
    # segment table name
    segment_table: list = field(default_factory=lambda: ["seg_A"])

    # Last computed depth
    last_computed_depth: int = 0

    # Number of times the tree has been repeated, for unrolling purposes
    tree_repetition: int = 1

    # Type of the tree
    tree_type: str = "main"  # "mcm" or "for_loop"
    
    def __repr__(self):
        state_to_special_str = [s.name if s != 0 else s for s in self.state_to_special]
        output = f"Name: {self.name}, tree_depth: {self.tree_depth}\n\t"
        output += f"tree_repetition: {self.tree_repetition}\n\t"
        output += f"tree_type: {self.tree_type}\n\t"
        output += f"visited: {self.visited},\n\t"
        output += f"special: {self.special},\n\t"
        output += f"probs: {self.probs}, \n\t"
        output += f"segment_table: {self.segment_table} \n\t"
        output += f"state_to_special: {state_to_special_str} \n\t"
        output += f"last_computed_depth: {self.last_computed_depth}"
        return output


def push_mcm_2_structure(state, mcm_count, position=None, op_type="mcm", new_state=None):
    """ Push mcm or for_loop structure to the traversal state"""
    
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

        # Adding a mcm after the for loop
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
    """ Expand the tree based on the repetition count and if the state is a for loop """
    repetitions = state.tree_repetition
    
    state.tree_repetition = 1
    state.tree_depth = state.tree_depth * repetitions     
    
    state.visited = state.visited * repetitions
    state.special = state.special * repetitions
    state.probs = state.probs * repetitions
    state.segment_table = state.segment_table * repetitions
    state.state_to_special = state.state_to_special * repetitions
    
    count = 0
    for i in state.state_to_special:
        
        if i != 0:
            if state.tree_type == "main":
                expand_repetition(i)
            elif state.tree_type == "for_loop":
                count += 1
                new_state = copy.deepcopy(i)
                new_state.name = new_state.name + "_rep" + str(count)
                state_to_special_index = state.state_to_special.index(i)
                
                state.state_to_special[state_to_special_index] = new_state
                expand_repetition(new_state)


def add_tree_to_global(state, global_visited, global_special, global_probs, global_segment_table, global_nested_states, depth=0):
    """ Add a traversal state to the global traversal state, handling for loops """
        
    for i in range(state.tree_repetition):
        for i in range(state.tree_depth):                
            if state.special[i] == "for_loop":
                add_tree_to_global(state.state_to_special[i], global_visited, global_special, global_probs, global_segment_table, global_nested_states, depth=depth+1)
                continue
            
            global_visited.append(state.visited[i])
            global_special.append(state.special[i])
            global_probs.append(state.probs[i])
            global_segment_table.append(state.segment_table[i])
            global_nested_states.append(depth)

def extract_segments(state, segment_list):
    for i in state.segment_table:
        
        if i == "for_loop":
            continue
        
        if i not in segment_list:
            segment_list.append(i)
            # for s in state.state_to_special:
            #     if s != 0:
            #         extract_segments(s, segment_list)
                    
        # add hash_ to the begining of the segment name in the state list 
        state.segment_table[state.segment_table.index(i)] = "hash_" + i
                    
    for s in state.state_to_special:
        if s != 0:
            extract_segments(s, segment_list)

            

if __name__ == "__main__":

    # Circuit Example
    
    #    Segment main_A
    #    mcm
    #    Segment main_B
    #   
    #    for loop start range (2)
    #        Segment FL_1_A
    #        mcm
    #        Segment FL_1_B
    # 
    #        for loop start range (2)
    #            Segment FL_2_A
    #            mcm
    #            Segment FL_2_B
    #
    #        Segment FL_1_D
    #
    #    Segment main_D
    #    mcm
    #    Segment main_E
    

    # Initialize the TraversalState | Segment A
    main_state = traversalState(
        name="main",
    )
    # Adding a mcm to main | Segment B
    push_mcm_2_structure(main_state, mcm_count=1)

    # Adding a for loop to main | for loop start | Segment FL_1_A and Segment main_D
    for_state_1 = traversalState(
        name="FL_1",
        tree_type="for_loop",
    )
    # Define iterations in the for loop state
    for_state_1.tree_repetition = 2
    push_mcm_2_structure(main_state, mcm_count=1, op_type="for_loop", new_state=for_state_1)
    # Adding mcm to for loop state | Segment FL_1_B
    push_mcm_2_structure(for_state_1, mcm_count=1)

    # Adding a nested for loop to FL_1 | for loop start | Segment FL_2_A and Segment FL_1_C
    for_state_2 = traversalState(
        name="FL_2",
        tree_type="for_loop",
    )
    # Define iterations in the nested for loop state
    for_state_2.tree_repetition = 2
    push_mcm_2_structure(for_state_1, mcm_count=1, op_type="for_loop", new_state=for_state_2)
    
    push_mcm_2_structure(for_state_2, mcm_count=1) # Segment FL_2_B
            
    # Adding another mcm to main
    push_mcm_2_structure(main_state, mcm_count=1) # Segment main_E
    # push_mcm_2_structure(main_state, mcm_count=1) # Segment main_F

    # Final state
    print_state(main_state)
    print("-"*100)
    print("-"*100)
    print("-"*100)
    
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    
    # Extract all segments in the circuit
    
    segment_list = []
        
    extract_segments(main_state, segment_list)
    print("Segments in the circuit:")
    for seg in segment_list:
        print(f" - {seg}")
    print("-"*100)
    print("-"*100)
    print("-"*100)
    
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    TT_CASE = "global"  # "global" or "nested"
    # TT_CASE = "nested"  # "global" or "nested"

    match TT_CASE:
        case "global":
            print("Using a single global traversal state")
            global_visited = []
            global_special = []
            global_probs = []
            global_segment_table = []
            global_state_to_special = []
            global_nested_states = []

            add_tree_to_global(main_state, global_visited, global_special, global_probs, global_segment_table, global_nested_states)

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
                print(f"Depth {i:>4}: visited={global_visited[i]}, special={global_special[i]}, probs={global_probs[i]}, segment_table={global_segment_table[i]+whitespace * global_nested_states[i]:>30}")

            # Case 1 : using single traversal though global state
            # tree_traversal(global_state, segment_list)

        case "nested":
            print("Using nested traversal states")
    
            expand_repetition(main_state)
            print_state(main_state)
            print("-"*100)
            
            # Case 2 : using nested traversal
            # tree_traversal(main_state, segment_list)
