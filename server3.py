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
# from control import *

ServerSideSocket = socket.socket()
ServerSideSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
host = '192.168.43.85'
# host = '192.168.50.226'
port = 6531
ThreadCount = 0
angle = 0.0
throttle = 0.0
finish = False
measured = False
# duration = 1
# mutex = allocate_lock()

try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))
print('Socket is listening..')
ServerSideSocket.listen(5)
def multi_threaded_client(connection):
    # global variables
    global throttle
    global angle
    global finish
    global measured

    #mbed measurements
    # dis = 100.0
    # right_dis = 100.0
    # left_dis = 100.0
    # acc_x = 0
    # acc_y = 0
    # acc_z = 0
    
    #measurement records
    last_dis_up = float("inf")
    last_dis_back = float("inf")
    # last_dis = 0
    last_acc_x = 0
    last_acc_y = 0
    last_acc_z = 0
    angle_record = [0 for i in range(3)]
    throttle_record = [0 for i in range(3)]
    
    
    #map
    node_index = 0
    route_set = route2
    route = route_set[0]
    car_pos = route_set[1]
    destination_index = len(route_set[0]) - 1

    #in a node
    roof_height = 30
    in_roof = False    #error correct about top back ultrasonic
    rotation_radius = 7

    max_throttle = 0.72
    angle_to_radius_ratio = 0.25

    #unused
    # global duration
    # throttle_record = 0.0
    # time_record = time.time()
    # dis_record = [0 for i in range(5)]
    # initial = True
    # initial_acc_x = 0
    # initial_acc_y = 0
    # initial_acc_z = 0
    # dis_left_record = [0 for i in range(5)]
    # dis_right_record = [0 for i in range(5)]

    while True:
        data = connection.recv(2048)
        req = data.decode('utf-8')
        print(data)
        # print(mutex.locked())
        if not data:
            break
        if req == "request for control":
            if finish:
                print("finish")
                throttle = -1
                response = {"angle": angle, "throttle": throttle}
            elif measured:
                # mutex.acquire()
                # if time.time() - time_record > duration:
                #     # duration = 0.8
                #     time_record = time.time()
                #     throttle_record = throttle
                # print("throttle: ", throttle_record, ", angle: ", angle)
                # response = {"angle": angle, "throttle": throttle_record}
                # mutex.release()
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
        #         # mutex.acquire()
        #         if time.time() - time_record > duration:
        #             time_record = time.time()
        #             throttle_record = throttle
        #             if dis > 15:
        #                 throttle_record = 0.6
        #                 # if abs(last_acc_x - acc_x) < 7 and abs(last_acc_y - acc_y) < 7 and abs(last_acc_z - acc_z) < 7:
        #                 #     throttle_record = 0.75
        #                 if abs(angle) == 1:
        #                     throttle_record = 0.7
        #         # if dis > 20:
        #         #     print("throttle: ", throttle_record, ", angle: ", angle)
        #         #     response = {"angle": angle, "throttle": throttle_record}
        #         # else:
        #         print("throttle: ", throttle_record, ", angle: ", angle)
        #         response = {"angle": angle, "throttle": throttle_record}
        #         # mutex.release()
        #     else:
        #         print("throttle: ", 0.0, ", angle: ", 0.0)
        #         response = {"angle": 0.0, "throttle": 0.0}
            
        #     connection.sendall(str.encode(json.dumps(response)))
        elif req[0:8] == "distance":
            measured = True
            if "distance" in req[8:]:
                req = req[:req.find("distance", 8)]
            # else:
            # mutex.acquire()
            dis_list = req.split(" ")[1:]
            dis = int(dis_list[0])
            dis_left = int(dis_list[1])
            dis_right = int(dis_list[2])
            dis_up = int(dis_list[3])
            print("=================")
            print("distance up:", dis_up)
            dis_back = int(dis_list[4])
            acc_x = float(dis_list[6])
            acc_y = float(dis_list[7])
            acc_z = float(dis_list[8])
            delta_acc = math.sqrt((acc_x - last_acc_x)**2 + (acc_y - last_acc_y)**2 + (acc_z - last_acc_z)**2)


            # dis = dis_record[0] * 0.05 + dis_record[1] * 0.1 + dis_record[2] * 0.15 + dis_record[3] * 0.2 + dis_record[4] * 0.5
            # dis_right = dis_right_record[0] * 0.05 + dis_right_record[1] * 0.1 + dis_right_record[2] * 0.15 + dis_right_record[3] * 0.2 + dis_right_record[4] * 0.5
            # dis_left = dis_left_record[0] * 0.05 + dis_left_record[1] * 0.1 + dis_left_record[2] * 0.15 + dis_left_record[3] * 0.2 + dis_left_record[4] * 0.5

            # # integral part
            # # dis_i = sum(dis_record) / len(dis_record)
            # dis_right_i = sum(dis_right_record)
            # dis_left_i = sum(dis_left_record)

            # # differential part
            # dis_right_d = dis_right - dis_right_record[-1]
            # dis_left_d = dis_left - dis_left_record[-1]

            # # update past errors
            # dis_record = dis_record[1:] + [dis]
            # dis_left_record = dis_left_record[1:] + [dis_left]
            # dis_right_record = dis_right_record[1:] + [dis_right]

            # pid part
            # dis = dis_i
            # dis_right = 0.8*dis_right + 0.05*dis_right_i + 0.2*dis_right_d
            # if dis_right < 0:
            #     dis_right = 0
            # dis_left = 0.8*dis_left + 0.05*dis_left_i + 0.2*dis_left_d
            # if dis_left < 0:
            #     dis_left = 0
            try:
                t = min(15 / delta_acc, max_throttle)
            except:
                t = max_throttle + 0.1 * abs(angle)
                # if abs(angle) == 1:
                #     t = 0.75
            # throttle = math.log(dis)
            # if throttle < 0:
            #     throttle = 0
            # if throttle > 0.5:
            #     throttle = 0.5
            
            try:
                right_angle = -10/dis_right
            except:
                right_angle = float("-inf")
            try:
                left_angle = 10/dis_left
            except:
                left_angle = float("inf")
            a = left_angle + right_angle
            if a > 1:
                a = 1
            if a < -1:
                a = -1
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

            if (dis_up < roof_height and dis_up != 0) or in_roof: # Turn
                if dis > 15: # if it stops
                    # throttle_record = 0.65
                    if delta_acc < 5:
                        try:
                            t = min(1, 1 / delta_acc + 0.5)
                        except:
                            t = 1
                    
                if dis_up >= roof_height and dis_back >= roof_height:
                    in_roof = False
                    continue
                if last_dis_up < roof_height or (last_dis_back < roof_height and in_roof):
                    if instruction == 0:
                        print("advance")
                        pass
                    elif instruction == 1 and dis_left > rotation_radius: # LEFT
                        print("left")
                        a = max(- angle_to_radius_ratio * (dis_left - rotation_radius), -1)
                    elif instruction == 2 and dis_right > rotation_radius: # RIGHT
                        print("right")
                        a = min(angle_to_radius_ratio * (dis_right - rotation_radius), 1)
                else:
                    if node_index == destination_index:
                        instruction = -1
                        finish = True
                        print("arrive final node")
                    else:
                        node = route[node_index]
                        print("=====================")
                        print("node:", node)
                        print("=====================")
                        node_index += 1
                        # node name
                        next_node = route[node_index]
                        next_node_pos = map2[node].index(next_node) # Search adjacency list
                        instruction = getDirection(car_pos, next_node_pos)
                        car_pos = next_node_pos
                        t = 0
                in_roof = True


            #         if abs(acc_x - initial_acc_x) < 50 and abs(acc_y - initial_acc_y) < 50 and abs(acc_z - initial_acc_z) < 50:
            #             throttle = 0.55
            #             if abs(angle) == 1:
            #                 throttle = 0.7
            #                 duration = 0.01


                    # if abs(acc_x - initial_acc_x) < 30 and abs(acc_y - initial_acc_y) < 30 and abs(acc_z - initial_acc_z) < 30:
                    #     throttle = 0.65

            # if initial:
            #     initial_acc_x = acc_x                
            #     initial_acc_y = acc_y                
            #     initial_acc_z = acc_z
            #     initial = False

            if dis_up != 0:
                last_dis_up = dis_up
            last_dis_back = dis_back
            last_acc_x = acc_x
            last_acc_y = acc_y
            last_acc_z = acc_z
            # last_dis = dis
            angle_record = angle_record[1:] + [a]
            throttle_record = throttle_record[1:] + [t]
            angle = sum(angle_record)/len(angle_record)
            throttle = sum(throttle_record)/len(throttle_record)
            print(throttle, angle)
            # mutex.release()
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
