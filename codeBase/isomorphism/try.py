from isomorphism import isomorphism_0, isomorphism_1
# test isomer
isomer1 = [
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [1,1,1,0,0,0,0,0,1,0],
    [0,0,0,1,0,0,0,1,0,1],
    [0,0,0,0,1,1,1,0,1,0],
]

isomorph1 = [
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,1,1,1,0,1,0],
    [0,0,1,0,0,0,0,1,0,1],
    [1,1,0,1,0,0,0,0,1,0],
]

isomer2 = [
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [1,1,0,0,0,0,0,0,1,0],
    [0,0,1,1,0,0,0,1,0,1],
    [0,0,0,0,1,1,1,0,1,0],
]

arr = [[0,6], [7,9]]

# Run both methods on the same structure with different labels: expected True.
print('isomorphism_0:', isomorphism_0(isomer1, isomorph1, arr))
print('isomorphism_1:', isomorphism_1(isomer1, isomorph1, arr))

# Run both methods on different structures: expected False.
print('isomer_0:', isomorphism_0(isomer1, isomer2, arr))
print('isomer_1:', isomorphism_1(isomer1, isomer2, arr))
