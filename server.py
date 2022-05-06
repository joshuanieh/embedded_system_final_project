#!/usr/bin/env python3
"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json
import util

class S(BaseHTTPRequestHandler):
    def __init__(self):
        self.__map = []
        self.__adjList = dict()

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))
    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), post_data.decode('utf-8'))

        self._set_response()
        response = {"angle": 0.2, "throttle": 0.5, "drive_mode": "user", "recording": False}
        response_json = json.dumps(response)
        self.wfile.write(bytes(response_json, 'utf-8'))
    
    def setMap(self, map):
        '''
        map grid represented as list of lists -> void
        0: not available
        1: roads
        2: destinations
        3: crossroads
        '''
        # map = [[0,0,2,0,0,0,0,2,0,0],
        #        [0,0,1,0,0,0,0,1,0,0],
        #        [2,1,3,1,1,3,1,3,3,1],
        #        [0,0,1,0,0,1,0,0,1,0],
        #        [0,0,1,0,0,1,0,0,1,0],
        #        [0,0,3,1,1,3,1,1,3,2],
        #        [0,0,1,0,0,1,0,0,1,0],
        #        [2,1,3,1,1,3,0,0,1,0],
        #        [0,0,1,0,0,1,0,0,1,0],
        #        [0,0,2,1,1,3,1,1,3,2]]
        n_row, n_col = len(map), len(map[0])
        for row in map:
            row.insert(0,0)
            row.append(0)
        self.__adjList = dict()
        map.insert(0, [0]*(n_col+2))
        map.append([0]*(n_row+2))
        self.__map = map
        for i in range(1, n_row+1):
            for j in range(1, n_col+1):
                self.__adjList[(i, j, map[i][j])] = []
                if map[i][j] != 0:
                    if map[i-1][j] != 0:
                        self.__adjList[(i, j, map[i][j])].append((i-1, j, map[i-1][j]))
                    if map[i+1][j] != 0:
                        self.__adjList[(i, j, map[i][j])].append((i+1, j, map[i+1][j]))
                    if map[i][j-1] != 0:
                        self.__adjList[(i, j, map[i][j])].append((i, j-1, map[i][j-1]))
                    if map[i][j+1] != 0:
                        self.__adjList[(i, j, map[i][j])].append((i, j+1, map[i][j+1]))
                if self.__adjList[(i, j, map[i][j])] == []:
                    self.__adjList.pop((i, j, map[i][j]))

    def getPath(self, start, destination):
        '''
        (x,y), (x,y) -> list of directions
        '''
        def manhattanDistance(p1, p2):
            return (p1[0] - p2[0]) + (p1[1] - p2[1])
        
        # (x,y) --> (x,y,map[x][y])
        start = (start[0], start[1], self.__map[start[0]][start[1]])
        destination = (destination[0], destination[1], self.__map[destination[0]][destination[1]])

        priorityQueue = util.PriorityQueue()
        # v = (x,y,map[x][y])
        # [(v, action, g(v), h(v))_1, (v, action, g(v), h(v))_2, (v, action, g(v), h(v))_3, ...]

        hasVisited = []
        currentVertex = start
        priorityQueue.push([(currentVertex, 'None', 0, manhattanDistance(currentVertex, destination))], 0 + manhattanDistance(currentVertex, destination))
        
        while True:
            vertices_actions_costs = priorityQueue.pop()
            vertices = [x[0] for x in vertices_actions_costs]
            cumulativeCosts = [x[2] for x in vertices_actions_costs]
            currentVertex = vertices[-1]
            currentCumulativeCost = cumulativeCosts[-1]
            if currentVertex == destination:
                break
            if currentVertex in hasVisited:
                continue
            for nextVertex in self.__adjList[currentVertex]:
                if nextVertex not in vertices:
                    if currentVertex[0] + 1 == nextVertex[0]:
                        action = 'South'
                    elif currentVertex[0] - 1 == nextVertex[0]:
                        action = 'North'
                    elif currentVertex[1] + 1 == nextVertex[1]:
                        action = 'East'
                    elif currentVertex[1] - 1 == nextVertex[1]:
                        action = 'West'
                    tmp = vertices_actions_costs[:]
                    tmp.append((nextVertex, action, currentCumulativeCost + 1, manhattanDistance(nextVertex, destination)))
                    priorityQueue.push(tmp, currentCumulativeCost + 1 + manhattanDistance(nextVertex, destination))
            hasVisited.append(currentVertex)
        return [x[1] for x in vertices_actions_costs][1:]
                    
def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('192.168.50.226', port)
    print(server_address)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
