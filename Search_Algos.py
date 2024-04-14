from graph1 import Graph, graph_1
import heapq
from asyncio import PriorityQueue
from queue import Queue

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


def ucs(graph, start, goal):
   
    frontier = [(0, start)] 
    explored = set()
    cost_so_far = {start: 0}
    came_from = {}

    while frontier:
        current_cost, current_node = heapq.heappop(frontier)

        if current_node == goal:
            path = [goal]
            while current_node != start:
                current_node = came_from[current_node]
                path.append(current_node)
            path.reverse()
            return path

        explored.add(current_node)

        for neighbor in graph.get_neighbors(current_node):
            new_cost = current_cost + graph.get_edge_weight(current_node, neighbor)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current_node

    return None



def bidirectional_search(graph, source, destination):
    source_parent = {}
    source_visited = set()
    source_queue = Queue()
    dest_parent = {}
    dest_visited = set()
    path = []
    dest_queue = Queue()

    source_queue.put(source)
    dest_queue.put(destination)
    
    if destination in graph.graph and source in graph.graph[destination]:
        return [source, destination]

    while True and not source_queue.empty() and not dest_queue.empty():
        selected = source_queue.get()
        source_visited.add(selected)
        connected = graph.get_neighbors(selected)

        if connected:
            for child in connected:
                if child not in source_parent:
                    source_parent[child] = selected
            
                if child not in source_queue.queue and child not in source_visited:
                    source_queue.put(child)
        

        selected_dest = dest_queue.get()
        dest_visited.add(selected_dest)
        connected_dest = graph.get_neighbors(selected_dest)
        

        if connected_dest:
            for child in connected_dest:
                if child not in dest_parent:
                    dest_parent[child] = selected_dest
                if child not in dest_queue.queue and child not in dest_visited:
                    dest_queue.put(child)

        for each in source_queue.queue:
            if each == source or each == destination:
                break
            if each in dest_queue.queue:
                path_dest = []
                current_dest = each
                while current_dest != destination:
                    path_dest.append(current_dest)
                    current_dest = dest_parent[current_dest]
                path_dest.append(destination)
                
                current = each
                while current != source:
                    if current not in path:
                        path.insert(0, current)
                        current = source_parent[current]
                
                path.insert(0, source)
            
                return path + path_dest



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
    "Bucharest": 0 
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
    "Bucharest": 0  
}
path7 = a_star_search(graph_1,"Arad", "Bucharest",heuristic, cost)
print(path1)
print(path2)
print(path3)
print(path4)
print(path5)
print(path6)
print(path7)


