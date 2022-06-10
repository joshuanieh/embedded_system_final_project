import socket
import os
from _thread import *
import json
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
    while True:
        data = connection.recv(2048)
        req = data.decode('utf-8')
        print(data)
        if not data:
            break
        if req == "request for control":
            print("throttle: ", throttle, ", angle: ", angle)
            response = {"angle": angle, "throttle": throttle}
            connection.sendall(str.encode(json.dumps(response)))
        if req == "request for start":
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

                throttle = dis/500
                if throttle > 0.5:
                    throttle = 0.5
                
                right_angle = dis_right
                left_angle = dis_left
                if right_angle == 5:
                    right_angle = 4.9
                if left_angle == 5:
                    left_angle = 4.9

                right_angle = 1/(right_angle - 5)
                if right_angle < 0 or right_angle > 1:
                    right_angle = -1
                else:
                    right_angle *= -1

                left_angle = 1/(left_angle - 5)
                if left_angle < 0 or left_angle > 1:
                    left_angle = 1
                else:
                    left_angle *= 1
                angle = left_angle + right_angle
    connection.close()
while True:
    Client, address = ServerSideSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(multi_threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
ServerSideSocket.close()
