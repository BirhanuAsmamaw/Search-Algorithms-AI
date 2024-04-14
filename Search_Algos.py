from graph1 import Graph
import heapq
from asyncio import PriorityQueue, Queue

graph_1 = Graph(directed=False)

nodes_data = {
    "Oradea": (47.0465005, 21.9189438),
    "Zerind": (46.622511, 21.517419),
    "Arad": (46.166667, 21.316667),
    "Timisoara": (45.759722, 21.23),
    "Lugoj": (45.68861, 21.90306),
    "Mehadia": (44.904114, 22.364516),
    "Drobeta": (44.636923, 22.659734),
    "Craiova": (44.333333, 23.816667000000052),
    "Sibiu": (45.792784, 24.152068999999983),
    "Rimnicu Vilcea": (45.099675, 24.369318),
    "Fagaras": (45.8416403, 24.9730954),
    "Pitesti": (44.860556, 24.867778000000044),
    "Giurgiu": (43.9037076, 25.9699265),
    "Bucharest": (44.439663, 26.096306),
    "Urziceni": (44.7165317, 26.641121),
    "Eforie": (44.058422, 28.633607),
    "Hirsova": (44.6833333, 27.9333333),
    "Vaslui": (46.640692, 27.727647),
    "Iasi": (47.156944, 27.590278000000012),
    "Neamt": (47.2, 26.3666667)
}

for node, (latitude, longitude) in nodes_data.items():
    graph_1.add_node(node, latitude, longitude)

graph_1.add_edge("Oradea", "Zerind", 71)
graph_1.add_edge("Zerind", "Arad", 75)
graph_1.add_edge("Arad", "Timisoara", 118)
graph_1.add_edge("Timisoara", "Lugoj", 111)
graph_1.add_edge("Lugoj", "Mehadia", 70)
graph_1.add_edge("Mehadia", "Drobeta", 75)
graph_1.add_edge("Drobeta", "Craiova", 120)
graph_1.add_edge("Craiova", "Rimnicu Vilcea", 146)
graph_1.add_edge("Rimnicu Vilcea", "Sibiu", 80)
graph_1.add_edge("Sibiu", "Fagaras", 99)
graph_1.add_edge("Fagaras", "Bucharest", 211)
graph_1.add_edge("Bucharest", "Pitesti", 101)
graph_1.add_edge("Pitesti", "Craiova", 138)
graph_1.add_edge("Bucharest", "Giurgiu", 90)
graph_1.add_edge("Bucharest", "Urziceni", 85)
graph_1.add_edge("Urziceni", "Vaslui", 142)
graph_1.add_edge("Vaslui", "Iasi", 92)
graph_1.add_edge("Iasi", "Neamt", 87)
graph_1.add_edge("Urziceni", "Hirsova", 98)
graph_1.add_edge("Hirsova", "Eforie", 86)
graph_1.add_edge("Sibiu", "Rimnicu Vilcea", 80)
graph_1.add_edge("Rimnicu Vilcea", "Pitesti", 97)
graph_1.add_edge("Rimnicu Vilcea", "Craiova", 146)
graph_1.add_edge("Sibiu", "Arad", 140)
graph_1.add_edge("Zerind", "Oradea", 71)


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

def iterative_deepening_dfs(graph, start_node, goal_node, max_depth=100):
    for depth in range(max_depth):
        result = dfs_limit(graph, start_node, goal_node, depth)
        if result is not None:
            return result
    return None

def dfs_limit(graph, start_node, goal_node, depth_limit):
    visited = set()
    stack = []
    stack.append((start_node, [start_node], 0))

    while stack:
        current_node, path, depth = stack.pop()
        if depth > depth_limit:
            continue
        if current_node == goal_node:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in reversed(list(graph.get_neighbors(current_node))):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))
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

def a_star_search(graph, start_node, goal_node, heuristic, cost):
    visited = set()
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic[start_node] + cost[start_node], start_node, [start_node], 0))

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

# Example usage:
path1 = dfs(graph_1,"Arad", "Bucharest")
path2 = bfs(graph_1,"Arad", "Bucharest")
path3 = ucs(graph_1,"Arad", "Bucharest")
path4 = bidirectional_search(graph_1,"Arad", "Bucharest")
heuristic = {
    "Oradea": 380,
    "Zerind": 374,
    "Arad": 366,  
    "Timisoara": 329,  
    "Lugoj": 244,  
    "Mehadia": 241,  
    "Drobeta": 242,  
    "Craiova": 160,  
    "Sibiu": 253,  
    "Rimnicu Vilcea": 193,  
    "Fagaras": 178,  
    "Pitesti": 98,  
    "Giurgiu": 77,  
    "Urziceni": 80,  
    "Eforie": 161,  
    "Hirsova": 151,  
    "Vaslui": 199,  
    "Iasi": 226,  
    "Neamt": 234,  
    "Bucharest": 0  # Add heuristic for Bucharest
}
path5 = greedy_search(graph_1,"Arad", "Bucharest", heuristic)
path6 = iterative_deepening_dfs(graph_1,"Arad", "Bucharest")
cost = {
    "Oradea": 0,
    "Zerind": 71,
    "Arad": 75,
    "Timisoara": 118,
    "Lugoj": 111,
    "Mehadia": 70,
    "Drobeta": 75,
    "Craiova": 120,
    "Sibiu": 140,
    "Rimnicu Vilcea": 80,
    "Fagaras": 99,
    "Pitesti": 97,
    "Giurgiu": 90,
    "Urziceni": 85,
    "Eforie": 86,
    "Hirsova": 98,
    "Vaslui": 142,
    "Iasi": 92,
    "Neamt": 87,
    "Bucharest": 0  # Add cost for Bucharest
}
path7 = a_star_search(graph_1,"Arad", "Bucharest",heuristic, cost)
print(path1)
print(path2)
print(path3)
print(path4)
print(path5)
print(path6)
print(path7)


