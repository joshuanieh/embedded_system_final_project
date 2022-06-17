# 0: North
# 1: South
# 2: West
# 3: East

ADVANCE = {(0, 0), (1, 1), (2, 2), (3, 3)}
RIGHT = {(0, 3), (1, 2), (2, 0), (3, 1)}
LEFT = {(0, 2), (1, 3), (2, 1), (3, 0)}

def getDirection(car_pos, next_node_pos):
    if (car_pos, next_node_pos) in ADVANCE:
        return 0 # (ADVANCE, car_pos_next)
    elif (car_pos, next_node_pos) in LEFT:
        return 1
    elif (car_pos, next_node_pos) in RIGHT:
        return 2
    else:
        raise Exception