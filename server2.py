#!/usr/bin/env python3

import socket
import json
import numpy as np
import matplotlib.pyplot as plot
HOST = '192.168.50.103' # IP address
PORT = 6531 # Port to listen on (use ports > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Starting server at: ", (HOST, PORT))
    conn, addr = s.accept()
    with conn:
        print("Connected at", addr)
        k = 0
        while True:
            data = conn.recv(1024).decode('utf-8', errors='ignore')
            #print(unicode(data, errors='replace'))
            print("Received from socket server:", data)
