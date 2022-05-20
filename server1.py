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
try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))
print('Socket is listening..')
ServerSideSocket.listen(5)
def multi_threaded_client(connection):
    global throttle
    global angle
    while True:
        data = connection.recv(2048)
        req = data.decode('utf-8')
        print(data)
        if not data:
            break
        if req == "request for control":
            print(throttle)
            response = {"angle": angle, "throttle": throttle}
            connection.sendall(str.encode(json.dumps(response)))
        if req[0:8] == "distance":
            if "distance" in req[8:]:
                continue
            else:
                dis_list = req.split(" ")[1:]
                throttle = int(dis_list[0])
                throttle = 2**throttle/320
                if throttle > 0.8:
                    throttle = 0.8
                angle = 1/(int(dis_list[1]) + 1)
                if angle > 1:
                    angle = -1
                else:
                    angle *= -1
    connection.close()
while True:
    Client, address = ServerSideSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(multi_threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
ServerSideSocket.close()
