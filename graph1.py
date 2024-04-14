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




g = Graph(directed=False)
g.add_edge(1, 2, weight=0.5)
g.add_edge(2, 3, weight=1.0)

# print("Undirected Graph:")
# print(g)

g_directed = Graph()
g_directed.add_edge(1, 2, weight=0.5)
g_directed.add_edge(2, 3, weight=1.0)

# print("Directed Graph:")
# print(g_directed)


# Loading the graph from the textbook

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

print("Graph 1:")
print(graph_1)




