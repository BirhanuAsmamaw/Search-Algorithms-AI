from graph1 import Graph, graph_1
import math
import numpy as np



# Degree Centrality
def degree_centrality(graph):
    centrality_scores = {}
    max_possible_degree = len(graph.graph) - 1 if graph.directed else 2 * (len(graph.graph) - 1)
    
    for node in graph.graph:
        neighbors = graph.get_neighbors(node)
        degree = len(neighbors)
        centrality_scores[node] = degree / max_possible_degree

    return centrality_scores


degree_scores = degree_centrality(graph_1)
print("Degree Centrality")
for node, score in degree_scores.items():
    print(f"{node}: {score}")


# Closeness Centrality
def closeness_centrality(graph):
    centrality_scores = {}
    for node in graph.graph:
        visited = set()
        queue = [(node, 0)]
        total_distance = 0
        while queue:
            current_node, distance = queue.pop(0)
            if current_node not in visited:
                visited.add(current_node)
                total_distance += distance
                neighbors = graph.get_neighbors(current_node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
        if len(visited) > 1:  
            centrality_scores[node] = (len(visited) - 1) / total_distance
        else:
            centrality_scores[node] = 0  
    return centrality_scores


# closeness_scores = closeness_centrality(graph_1)
# for node, score in closeness_scores.items():
#     print(f"{node}: {score}")


# Eigenvector Centrality
def adjacency_matrix(graph):
    num_nodes = len(graph.graph)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    node_index_map = {node: i for i, node in enumerate(graph.graph)}

    for node in graph.graph:
        neighbors = graph.get_neighbors(node)
        for neighbor in neighbors:
            j = node_index_map[neighbor]
            adj_matrix[node_index_map[node]][j] = 1

    return adj_matrix


def eigenvector_centrality(graph, max_iter=100, tol=1.0e-6):
    # Initialize eigenvector scores
    eigenvector_scores = {node: 1.0 for node in graph.graph}
    num_nodes = len(graph.graph)

    # Calculate the adjacency matrix
    adj_matrix = adjacency_matrix(graph)

    # Normalize the adjacency matrix
    row_sums = adj_matrix.sum(axis=1)
    adj_matrix = adj_matrix / row_sums[:, np.newaxis]

    # Convert eigenvector scores to a NumPy array
    initial_scores_array = np.array([eigenvector_scores[node] for node in graph.graph])

    # Power iteration method to find eigenvector centrality
    for _ in range(max_iter):
        prev_scores = eigenvector_scores.copy()
        eigenvector_scores_array = np.dot(adj_matrix.T, initial_scores_array)
        eigenvector_scores_array /= np.linalg.norm(eigenvector_scores_array, ord=2)

        # Convert eigenvector scores array back to dictionary
        eigenvector_scores = {node: eigenvector_scores_array[i] for i, node in enumerate(graph.graph)}

        # Check for convergence
        if np.linalg.norm(np.array(list(eigenvector_scores.values())) - np.array(list(prev_scores.values())), ord=2) < tol:
            break

    return eigenvector_scores


# eigenvector_scores = eigenvector_centrality(graph_1)
# for node, score in eigenvector_scores.items():
#     print(f"{node}: {score}")


# katz Centrality

def katz_centrality(graph, alpha=0.1, beta=1, max_iter=100, tol=1.0e-6):
    num_nodes = len(graph.graph)
    adj_matrix = adjacency_matrix(graph)
    centrality_scores = {node: 0 for node in graph.graph}

    # Power iteration method to find Katz centrality
    for _ in range(max_iter):
        prev_scores = centrality_scores.copy()
        centrality_scores = {node: alpha for node in graph.graph}
        for node in graph.graph:
            neighbors = graph.get_neighbors(node)
            for neighbor in neighbors:
                centrality_scores[node] += beta * prev_scores[neighbor]

        # Normalize centrality scores
        norm_factor = max(centrality_scores.values())
        centrality_scores = {node: score / norm_factor for node, score in centrality_scores.items()}

        # Check for convergence
        if np.linalg.norm(np.array(list(centrality_scores.values())) - np.array(list(prev_scores.values())), ord=2) < tol:
            break

    return centrality_scores


# katz_scores = katz_centrality(graph_1)
# for node, score in katz_scores.items():
#     print(f"{node}: {score}")



#PageRank

def page_rank(graph, damping_factor=0.85, max_iter=100, tol=1.0e-6):
    num_nodes = len(graph.graph)
    adj_matrix = adjacency_matrix(graph)
    centrality_scores = {node: 1 / num_nodes for node in graph.graph}

    # Power iteration method to find PageRank
    for _ in range(max_iter):
        prev_scores = centrality_scores.copy()
        centrality_scores = {node: (1 - damping_factor) / num_nodes for node in graph.graph}
        for node in graph.graph:
            neighbors = graph.get_neighbors(node)
            if len(neighbors) > 0:
                for neighbor in neighbors:
                    centrality_scores[neighbor] += damping_factor * prev_scores[node] / len(neighbors)

        # Check for convergence
        if np.linalg.norm(np.array(list(centrality_scores.values())) - np.array(list(prev_scores.values())), ord=2) < tol:
            break

    return centrality_scores


# pagerank_scores = page_rank(graph_1)
# for node, score in pagerank_scores.items():
#     print(f"{node}: {score}")


# Betweenness Centrality
def betweenness_centrality(graph):
    centrality_scores = {node: 0 for node in graph.graph}

    for node in graph.graph:
        stack = []
        predecessors = {n: [] for n in graph.graph}
        sigma = {n: 0 for n in graph.graph}
        sigma[node] = 1
        distance = {n: -1 for n in graph.graph}
        distance[node] = 0
        queue = [node]

        while queue:
            v = queue.pop(0)
            stack.append(v)

            for neighbor in graph.get_neighbors(v):
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[v] + 1

                if distance[neighbor] == distance[v] + 1:
                    sigma[neighbor] += sigma[v]
                    predecessors[neighbor].append(v)

        delta = {n: 0 for n in graph.graph}

        while stack:
            w = stack.pop()
            for predecessor in predecessors[w]:
                delta[predecessor] += (sigma[predecessor] / sigma[w]) * (1 + delta[w])
            if w != node:
                centrality_scores[w] += delta[w]

    # Normalize betweenness centrality scores
    num_pairs = (len(graph.graph) - 1) * (len(graph.graph) - 2) / 2 if graph.directed else (len(graph.graph) - 1) * (len(graph.graph)) / 2
    for node in centrality_scores:
        centrality_scores[node] /= num_pairs

    return centrality_scores


# betweenness_scores = betweenness_centrality(graph_1)
# for node, score in betweenness_scores.items():
#     print(f"{node}: {score}")
