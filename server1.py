from dis import Instruction
import socket
import os
from _thread import *
import json
from map import * 
from steer import * 

ServerSideSocket = socket.socket()
host = '192.168.50.226'
port = 6531
ThreadCount = 0
angle = 0.0
throttle = 0.0
dis = 100.0
right_dis = 100.0
left_dis = 100.0

try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))
print('Socket is listening..')
ServerSideSocket.listen(5)
def multi_threaded_client(connection):
    global throttle
    global angle
    global dis
    global right_dis
    global left_dis
    last_dis_up = 1201
    node_index = 0
    route = route1[0]
    car_pos = route1[1]
    destination_index = len(route1[0]) - 1
    finish = False
    
    while True:
        data = connection.recv(2048)
        req = data.decode('utf-8')
        print(data)
        if not data:
            break
        if req == "request for control":
            if finish:
                throttle = -1
                print("finish")
            else:
                print("throttle: ", throttle, ", angle: ", angle)
            response = {"angle": angle, "throttle": throttle}
            connection.sendall(str.encode(json.dumps(response)))
        if req == "request for start":
            if finish:
                throttle = -1
                print("finish")
            else:
                print("throttle: ", 0.7, ", angle: ", angle)
            response = {"angle": angle, "throttle": 0.7}
            if dis < 50:
                response = {"angle": angle, "throttle": throttle}
            connection.sendall(str.encode(json.dumps(response)))
        if req[0:8] == "distance":
            if "distance" in req[8:]:
                continue
            else:
                dis_list = req.split(" ")[1:]
                dis = int(dis_list[0])
                dis_left = int(dis_list[1])
                dis_right = int(dis_list[2])
                dis_up = int(dis_list[3])

                throttle = dis/500
                if throttle > 0.5:
                    throttle = 0.5
                
                right_angle = dis_right
                left_angle = dis_left
                if dis_right == 5:
                    dis_right = 4.9
                if dis_left == 5:
                    dis_left = 4.9

                right_angle = 1/(dis_right - 5)
                if right_angle < 0 or right_angle > 1:
                    right_angle = -1
                else:
                    right_angle *= -1

                left_angle = 1/(dis_left - 5)
                if left_angle < 0 or left_angle > 1:
                    left_angle = 1
                else:
                    left_angle *= 1
                angle = left_angle + right_angle

                if dis_up < 50: # Turn
                    if dis_up - last_dis_up < 3:
                        if instruction == 0:
                            pass
                        elif instruction == 1 and dis_left > 3: # LEFT
                            angle = -1
                        elif instruction == 2 and dis_right > 3: # RIGHT
                            angle = 1
                    else:
                        if node_index == destination_index:
                            finish = True # finish
                        else:
                            node = route[node_index]
                            node_index += 1
                            # node name
                            next_node = route[node_index]
                            next_node_pos = map1[node].index(next_node) # Search adjacency list
                            instruction = getDirection(car_pos, next_node_pos)
                            if instruction == 0:
                                pass
                            elif instruction == 1 and dis_left > 3: # LEFT
                                angle = -1
                            elif instruction == 2 and dis_right > 3: # RIGHT
                                angle = 1
                            car_pos = next_node_pos
    connection.close()
while True:
    Client, address = ServerSideSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(multi_threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
ServerSideSocket.close()
