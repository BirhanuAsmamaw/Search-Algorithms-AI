from asyncio import Queue as AsyncQueue

class Graph:
    def __init__(self, directed=True):
        self.graph = {}
        self.coordinates = {}
        self.directed = directed

    def add_node(self, node, latitude=None, longitude=None):
        if node not in self.graph:
            self.graph[node] = {}
            self.coordinates[node] = (latitude, longitude)

    def add_edge(self, node1, node2, weight=None):
        self.add_node(node1)
        self.add_node(node2)

        if not self.directed:
            self.graph[node1][node2] = weight
            self.graph[node2][node1] = weight
        else:
            self.graph[node1][node2] = weight

    def delete_node(self, node):
        if node in self.graph:
            del self.graph[node]
            del self.coordinates[node]
            for n in self.graph:
                if node in self.graph[n]:
                    del self.graph[n][node]

    def delete_edge(self, node1, node2):
        if node1 in self.graph and node2 in self.graph[node1]:
            del self.graph[node1][node2]
            if not self.directed and node2 in self.graph and node1 in self.graph[node2]:
                del self.graph[node2][node1]

    def get_neighbors(self, node):
        if node in self.graph:
            return self.graph[node].keys()
        else:
            return []

    def get_edge_weight(self, node1, node2):
        if node1 in self.graph and node2 in self.graph[node1]:
            return self.graph[node1][node2]
        else:
            return None

    def __str__(self):
        output = ""
        for node in self.graph:
            output += f"{node}: {self.graph[node]}\n"
        return output

import random
from asyncio import PriorityQueue, Queue
import heapq
import time

def generate_random_graph(n, p):
    graph = Graph(directed=False)
    nodes = [f"Node_{i}" for i in range(n)]
    
    # Generate coordinates for nodes
    coordinates = {node: (random.uniform(0, 100), random.uniform(0, 100)) for node in nodes}
    
    # Add nodes to the graph
    for node, (x, y) in coordinates.items():
        graph.add_node(node, latitude=x, longitude=y)
    
    # Connect nodes randomly based on probability p
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                weight = random.randint(1, 100)  # Random edge weight
                graph.add_edge(nodes[i], nodes[j], weight)
    
    return graph

async def bidirectional_search(graph, start_node, goal_node):
    forward_visited = set()
    backward_visited = set()
    forward_queue = AsyncQueue()
    backward_queue = AsyncQueue()
    forward_queue.put_nowait((start_node, [start_node]))
    backward_queue.put_nowait((goal_node, [goal_node]))

    while not forward_queue.empty() and not backward_queue.empty():
        forward_current_node, forward_path = forward_queue.get_nowait()
        backward_current_node, backward_path = backward_queue.get_nowait()

        if forward_current_node in backward_visited:
            intersection_node = forward_current_node
            backward_path.reverse()
            return forward_path + backward_path[1:]

        if forward_current_node not in forward_visited:
            forward_visited.add(forward_current_node)
            for neighbor in graph.get_neighbors(forward_current_node):
                if neighbor not in forward_visited:
                    forward_queue.put_nowait((neighbor, forward_path + [neighbor]))

        if backward_current_node not in backward_visited:
            backward_visited.add(backward_current_node)
            for neighbor in graph.get_neighbors(backward_current_node):
                if neighbor not in backward_visited:
                    backward_queue.put_nowait((neighbor, backward_path + [neighbor]))

    return None

def dfs(graph, start_node, goal_node):
    visited = set()
    stack = []
    stack.append((start_node, [start_node]))

    while stack:
        current_node, path = stack.pop()
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in reversed(list(graph.get_neighbors(current_node))):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None

def bfs(graph, start_node, goal_node):
    visited = set()
    queue = []
    queue.append((start_node, [start_node]))

    while queue:
        current_node, path = queue.pop(0)
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None

def ucs(graph, start_node, goal_node):
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start_node, [start_node]))
    while not pq.empty():
        cost, current_node, path = pq.get()
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    new_cost = cost + graph.get_edge_weight(current_node, neighbor)
                    pq.put((new_cost, neighbor, path + [neighbor]))
    return None

def greedy_search(graph, start_node, goal_node, heuristic):
    visited = set()
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic[start_node], start_node, [start_node]))

    while priority_queue:
        _, current_node, path = heapq.heappop(priority_queue)
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (heuristic[neighbor], neighbor, path + [neighbor]))

    return None

def a_star_search(graph, start_node, goal_node, heuristic):
    visited = set()
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic[start_node], start_node, [start_node], 0))

    while priority_queue:
        _, current_node, path, current_cost = heapq.heappop(priority_queue)
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                new_cost = current_cost + graph.get_edge_weight(current_node, neighbor)
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (new_cost + heuristic[neighbor], neighbor, path + [neighbor], new_cost))

    return None


algorithm_functions = [dfs, bfs, ucs, bidirectional_search, greedy_search, a_star_search]
algorithm_names = ["DFS", "BFS", "UCS", "Bidirectional Search", "Greedy Search", "A* Search"]

results = []

graph_settings = []
for n in [10, 20, 30, 40]:
    for p in [0.2, 0.4, 0.6, 0.8]:
        graph = generate_random_graph(n, p)
        graph_settings.append((n, p, graph))

for n, p, graph in graph_settings:
    random_nodes = {f"Node_{random.randint(0, n - 1)}" for _ in range(10)}
    heuristic = {node: random.randint(1, 100) for node in graph.graph.keys()}  # Dummy heuristic
    for algorithm, algorithm_name in zip(algorithm_functions, algorithm_names):
        total_time = 0
        for _ in range(5):
            start_time = time.time()
            for node1 in random_nodes:
                for node2 in random_nodes:
                    if node1 != node2:
                        if algorithm_name in ["Greedy Search", "A* Search"]:  # Provide heuristic for Greedy Search and A* Search
                            algorithm(graph, node1, node2, heuristic)
                        else:
                            algorithm(graph, node1, node2)  # Skip heuristic for other algorithms
            total_time += (time.time() - start_time)
        average_time = total_time / 5
        results.append((algorithm_name, n, p, average_time))

# Print results
for result in results:
    print(result)
