from dis import Instruction
import socket
import os
from _thread import *
import json
from map import * 
from steer import * 
from arrowDetect import arrowDetect
import time
import math

ServerSideSocket = socket.socket()
ServerSideSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
host = '192.168.43.85'
# host = '192.168.50.226'
port = 6531
ThreadCount = 0
angle = 0.0
throttle = 0.0
dis = 100.0
right_dis = 100.0
left_dis = 100.0
finish = False
measured = False

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
    global finish
    global measured
    last_dis_up = 1201
    node_index = 0
    route = route2[0]
    car_pos = route2[1]
    destination_index = len(route2[0]) - 1
    last_dis = 1201
    last_dis_back = 1201
    upset = False
    # throttle_record = 0.0
    # time_record = time.time()
    roof_height = 20
    duration = 0.8
    rotation_radius = 7
    dis_record = [0 for i in range(5)]
    dis_left_record = [0 for i in range(5)]
    dis_right_record = [0 for i in range(5)]
    acc_x = 0
    acc_y = 0
    acc_z = 0
    last_acc_x = 0
    last_acc_y = 0
    last_acc_z = 0
    initial = True
    initial_acc_x = 0
    initial_acc_y = 0
    initial_acc_z = 0

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
                response = {"angle": angle, "throttle": throttle}
            elif measured:
                # if time.time() - time_record > duration:
                    # time_record = time.time()
                    # throttle_record = throttle
                # print("throttle: ", throttle_record, ", angle: ", angle)
                # response = {"angle": angle, "throttle": throttle_record}
                print("throttle: ", throttle, ", angle: ", angle)
                response = {"angle": angle, "throttle": throttle}

            else:
                print("throttle: ", 0.0, ", angle: ", 0.0)
                response = {"angle": 0.0, "throttle": 0.0}

            connection.sendall(str.encode(json.dumps(response)))
        # elif req == "request for start":
        #     if finish:
        #         throttle = -1
        #         print("finish")
        #         response = {"angle": angle, "throttle": throttle}
        #     elif measured:
        #         if time.time() - time_record > duration:
        #             time_record = time.time()
        #             throttle_record = throttle
                    # if dis > 15:
                    #     throttle_record = 0.65
                    #     if abs(last_acc_x - acc_x) < 7 and abs(last_acc_y - acc_y) < 7 and abs(last_acc_z - acc_z) < 7:
                    #         throttle_record = 0.7
                # if dis > 20:
                #     print("throttle: ", 0.7, ", angle: ", angle)
                #     response = {"angle": angle, "throttle": 0.7}
                # else:
            #     print("throttle: ", throttle_record, ", angle: ", angle)
            #     response = {"angle": angle, "throttle": throttle_record}
            # else:
            #     print("throttle: ", 0.0, ", angle: ", 0.0)
            #     response = {"angle": 0.0, "throttle": 0.0}
            
            # connection.sendall(str.encode(json.dumps(response)))
        elif req[0:8] == "distance":
            measured = True
            if "distance" in req[8:]:
                continue
            else:
                dis_list = req.split(" ")[1:]
                dis = int(dis_list[0])
                dis_left = int(dis_list[1])
                dis_right = int(dis_list[2])
                dis_up = int(dis_list[3])
                dis_back = int(dis_list[4])
                last_acc_x = acc_x
                last_acc_y = acc_y
                last_acc_z = acc_z
                acc_x = float(dis_list[6])
                acc_y = float(dis_list[7])
                acc_z = float(dis_list[8])

                # dis = dis_record[0] * 0.05 + dis_record[1] * 0.1 + dis_record[2] * 0.15 + dis_record[3] * 0.2 + dis_record[4] * 0.5
                # dis_right = dis_right_record[0] * 0.05 + dis_right_record[1] * 0.1 + dis_right_record[2] * 0.15 + dis_right_record[3] * 0.2 + dis_right_record[4] * 0.5
                # dis_left = dis_left_record[0] * 0.05 + dis_left_record[1] * 0.1 + dis_left_record[2] * 0.15 + dis_left_record[3] * 0.2 + dis_left_record[4] * 0.5

                # integral part
                # dis_i = sum(dis_record) / len(dis_record)
                dis_right_i = sum(dis_right_record)
                dis_left_i = sum(dis_left_record)

                # differential part
                dis_right_d = dis_right - dis_right_record[-1]
                dis_left_d = dis_left - dis_left_record[-1]

                # update past errors
                dis_record = dis_record[1:] + [dis]
                dis_left_record = dis_left_record[1:] + [dis_left]
                dis_right_record = dis_right_record[1:] + [dis_right]

                # pid part
                # dis = dis_i
                dis_right = 0.8*dis_right + 0.05*dis_right_i + 0.2*dis_right_d
                if dis_right < 0:
                    dis_right = 0
                dis_left = 0.8*dis_left + 0.05*dis_left_i + 0.2*dis_left_d
                if dis_left < 0:
                    dis_left = 0

                throttle = dis/500
                if throttle > 0.4:
                    throttle = 0.4
                
                try:
                    right_angle = -50/dis_right
                except:
                    right_angle = float("-inf")
                try:
                    left_angle = 50/dis_left
                except:
                    left_angle = float("inf")
                # try:
                #     right_angle = 1/(dis_right - 6.5)
                # except:
                #     right_angle = -1
                # if right_angle < 0 or right_angle > 1:
                #     right_angle = -1
                # else:
                #     right_angle *= -1

                # try:
                #     left_angle = 1/(dis_left - 6.5)
                # except:
                #     left_angle = -1
                # if left_angle < 0 or left_angle > 1:
                #     left_angle = 1
                # else:
                #     left_angle *= 1
                angle = left_angle + right_angle
                if angle > 1:
                    angle = 1
                if angle < -1:
                    angle = -1

                if (dis_up < roof_height and dis_up != 0) or upset: # Turn
                    if dis_up >= roof_height and dis_back >= roof_height:
                        upset = False
                        continue
                    # if dis > 15 and dis == last_dis:
                    #     throttle = 0.6
                    if last_dis_up < roof_height or (last_dis_back < roof_height and upset):
                        if instruction == 0:
                            print("advance")
                            pass
                        elif instruction == 1 and dis_left > rotation_radius: # LEFT
                            print("left")
                            angle = -1
                        elif instruction == 2 and dis_right > rotation_radius: # RIGHT
                            print("right")
                            angle = 1
                    else:
                        if node_index == destination_index:
                            instruction = -1
                            finish = True
                            print("arrive final node")
                        else:
                            node = route[node_index]
                            print(node)
                            node_index += 1
                            # node name
                            next_node = route[node_index]
                            next_node_pos = map2[node].index(next_node) # Search adjacency list
                            instruction = getDirection(car_pos, next_node_pos)
                            if instruction == 0:
                                print("advance")
                                pass
                            elif instruction == 1 and dis_left > rotation_radius: # LEFT
                                print("left")
                                angle = -1
                            elif instruction == 2 and dis_right > rotation_radius: # RIGHT
                                print("right")
                                angle = 1

                            car_pos = next_node_pos
                    upset = True

                if dis > 15:
                        # throttle_record = 0.65
                        if abs(acc_x - initial_acc_x) < 70 and abs(acc_y - initial_acc_y) < 70 and abs(acc_z - initial_acc_z) < 70:
                            throttle = 0.55

                        if abs(acc_x - initial_acc_x) < 50 and abs(acc_y - initial_acc_y) < 50 and abs(acc_z - initial_acc_z) < 50:
                            throttle = 0.65

                        if abs(acc_x - initial_acc_x) < 30 and abs(acc_y - initial_acc_y) < 30 and abs(acc_z - initial_acc_z) < 30:
                            throttle = 0.75

                if initial:
                    initial_acc_x = acc_x                
                    initial_acc_y = acc_y                
                    initial_acc_z = acc_z
                    initial = False

                if dis_up != 0:
                    last_dis_up = dis_up
                last_dis_back = dis_back
                last_dis = dis
        elif req[0:2] == "NFC":
            finish = True
    connection.close()
while True:
    Client, address = ServerSideSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(multi_threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
ServerSideSocket.close()
