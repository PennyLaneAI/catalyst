"""
Does the tape need to be aware of the scope?
Could we just split out the tapes based on the op?
What's the most understandable and maintainable?

tape = [(0, 3, 1), (0, 2, 1), true, false, true, (0, 2, 1), true, false, true, (0, 2, 1), true, false, true, (0, 4, 0)]

for i from 0 to 3:
    for j from 0 to 2:
        if i is even:
            RX(thetas[i])

for k from 0 to 4:
    RY(thetas[k])

# Backwards:

for k' from 4 to 0:
    RY(thetas[k'])

for i' from 3 to 0:
    for j' from 4 to 0:
        if i' is even

It's a recursive procedure:
def reverse_block(block, tape):
    for operation in reversed(block.operations):
        if forOp = dyn_cast<scf::ForOp>(operation):
            
"""

"""
tape = [(0, n), (0, 0), (0, 1), ..., (0, n - 1)]
Triangular case:
for i from 0 to n:
    for j from 0 to i:
        print(i, j)

# What if you pulled the value directly?
# LAGrad does this, works for some for loops but not for while loops.
# Enzyme uses a recursive tape structure
# Tangent looks at attributes? I think it annotates the values with each other
for i from n to 0:
    for j from i to 0:
        print(i, j)
"""
n = 4
i = 0
# We don't know how big the tape is in practice
# Need to go to the last member of the scope

# We can use the Operation * as keys to a compile-time dictionary where the values are arraylists.
tape = [4, (0, 1, 2, 3)]
while i < n:
    j = 0
    while j < i:
        print(i, j)
        j += 1
    i += 1

# for i in range(0, 3):
#     for j in range(0, 2):
#         if i % 2 == 0:
#             print(i, j)

# print("Reversed:")
# for i in reversed(range(0, 3)):
#     for j in reversed(range(0, 2)):
#         if i % 2 == 0:
#             print(i, j)




"""
tapes: {
    0: [(0, n, 1)], -> (n - 1, -1, -1)
    1: [True, False, True...]
    2: [5, 6, 7, ... 4]
}
"""
