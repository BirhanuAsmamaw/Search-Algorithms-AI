from Centralities import *
from graph1 import Graph, graph_1

degree_centrality = degree_centrality(graph_1)
closeness_centrality = closeness_centrality(graph_1)
betweenness_centrality = betweenness_centrality(graph_1)
eigenvector_centrality = eigenvector_centrality(graph_1)
katz_centrality = katz_centrality(graph_1)
page_rank = page_rank(graph_1)

print("Degree Centrality")
print([(city, centrality) for city, centrality in degree_centrality.items()], end="\n\n")

print("Closeness Centrality")
print([(city, centrality) for city, centrality in closeness_centrality.items()], end="\n\n")

print("Betweenness Centrality")
print([(city, centrality) for city, centrality in betweenness_centrality.items()], end="\n\n")

print("Eigenvector Centrality")
print([(city, centrality) for city, centrality in eigenvector_centrality.items()], end="\n\n")

print("Katz Centrality")
print([(city, centrality) for city, centrality in katz_centrality.items()], end="\n\n")

print("Page Rank")
print([(city, centrality) for city, centrality in page_rank.items()], end="\n\n")
