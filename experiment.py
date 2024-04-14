import random
import time
import statistics
import heapq
from queue import Queue

# Define the Graph class
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


# Define the search algorithms
def dfs(graph, start_node, goal_node):
    visited = set()
    stack = [(start_node, [start_node])]
    while stack:
        current_node, path = stack.pop()
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None

def bfs(graph, start_node, goal_node):
    visited = set()
    queue = [(start_node, [start_node])]
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
    pq = []
    heapq.heappush(pq, (0, start_node, [start_node]))
    while pq:
        cost, current_node, path = heapq.heappop(pq)
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    new_cost = cost + graph.get_edge_weight(current_node, neighbor)
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
    return None

def bidirectional_search(graph, start_node, goal_node):
    forward_visited = set()
    backward_visited = set()
    forward_queue = Queue()
    backward_queue = Queue()
    forward_queue.put((start_node, [start_node]))
    backward_queue.put((goal_node, [goal_node]))

    while not forward_queue.empty() and not backward_queue.empty():
        forward_current_node, forward_path = forward_queue.get()
        backward_current_node, backward_path = backward_queue.get()

        if forward_current_node in backward_visited:
            intersection_node = forward_current_node
            backward_path.reverse()
            return forward_path + backward_path[1:]

        if forward_current_node not in forward_visited:
            forward_visited.add(forward_current_node)
            for neighbor in graph.get_neighbors(forward_current_node):
                if neighbor not in forward_visited:
                    forward_queue.put((neighbor, forward_path + [neighbor]))

        if backward_current_node not in backward_visited:
            backward_visited.add(backward_current_node)
            for neighbor in graph.get_neighbors(backward_current_node):
                if neighbor not in backward_visited:
                    backward_queue.put((neighbor, backward_path + [neighbor]))

    return None

def greedy_search(graph, start_node, goal_node, heuristic):
    visited = set()
    pq = []
    heapq.heappush(pq, (heuristic[start_node], start_node, [start_node]))
    while pq:
        _, current_node, path = heapq.heappop(pq)
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    heapq.heappush(pq, (heuristic[neighbor], neighbor, path + [neighbor]))
    return None

def iterative_deepening_dfs(graph, start_node, goal_node):
    depth_limit = 0
    while True:
        result = depth_limited_dfs(graph, start_node, goal_node, depth_limit)
        if result is not None:
            return result
        depth_limit += 1

def depth_limited_dfs(graph, start_node, goal_node, depth_limit):
    visited = set()
    stack = [(start_node, [start_node], 0)]
    while stack:
        current_node, path, depth = stack.pop()
        if depth > depth_limit:
            continue
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))
    return None

def a_star_search(graph, start_node, goal_node, heuristic, cost):
    visited = set()
    pq = []
    heapq.heappush(pq, (0 + heuristic[start_node], start_node, [start_node]))
    while pq:
        _, current_node, path = heapq.heappop(pq)
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get_neighbors(current_node):
                if neighbor not in visited:
                    new_cost = cost[current_node] + graph.get_edge_weight(current_node, neighbor)
                    heapq.heappush(pq, (new_cost + heuristic[neighbor], neighbor, path + [neighbor]))
    return None

# Function to generate random graphs
def generate_random_graph(n, p):
    graph = Graph(directed=False)
    nodes_data = {}
    for i in range(n):
        latitude = random.uniform(0, 100)
        longitude = random.uniform(0, 100)
        node_name = f"Node_{i}"
        nodes_data[node_name] = (latitude, longitude)
    for node, (latitude, longitude) in nodes_data.items():
        graph.add_node(node, latitude, longitude)
    for node1 in nodes_data.keys():
        for node2 in nodes_data.keys():
            if node1 != node2 and random.random() < p:
                weight = random.uniform(1, 100)  # Random edge weight
                graph.add_edge(node1, node2, weight)
    return graph

# Function to randomly select 10 nodes from a graph
def select_random_nodes(graph):
    return random.sample(list(graph.graph.keys()), 10)

# Function to run experiments and benchmark the algorithms
def run_experiments():
    num_trials = 10
    results = {}

    for n in [10, 20, 30, 40]:
        for p in [0.2, 0.4, 0.6, 0.8]:
            avg_runtimes = {}
            for algorithm in [dfs, bfs, iterative_deepening_dfs]:
                runtimes = []
                for _ in range(num_trials):
                    # Generate random graph
                    graph = generate_random_graph(n, p)
                    # Randomly select 10 nodes
                    random_nodes = select_random_nodes(graph)
                    # Measure runtime
                    start_time = time.time()
                    for i in range(len(random_nodes)):
                        for j in range(i+1, len(random_nodes)):
                            algorithm(graph, random_nodes[i], random_nodes[j])
                    end_time = time.time()
                    runtime = end_time - start_time
                    runtimes.append(runtime)
                avg_runtime = statistics.mean(runtimes)
                avg_runtimes[algorithm.__name__] = avg_runtime
            results[(n, p)] = avg_runtimes

    return results

# Run experiments
experiment_results = run_experiments()

# Print results
for (n, p), avg_runtimes in experiment_results.items():
    print(f"Graph with {n} nodes and edge probability {p}:")
    for algorithm, avg_runtime in avg_runtimes.items():
        print(f"{algorithm}: {avg_runtime} seconds")
