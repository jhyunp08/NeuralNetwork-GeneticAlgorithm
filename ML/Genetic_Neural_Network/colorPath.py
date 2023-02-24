import heapq
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, delta_e_cie2000, delta_e_cmc
from colormath.color_conversions import convert_color
import numpy

# patch discarded method asscalar of numpy, since color diff is dependent on this
def patch_asscalar(a):
    return a.item()
setattr(numpy, "asscalar", patch_asscalar)


def build_color_graph(colors, color_difference):
    graph = {}
    for i in range(len(colors)):
        graph[i] = {}
        for j in range(len(colors)):
            if i != j:
                diff = color_difference(colors[i], colors[j])
                graph[i][j] = diff
    return graph

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    
    while heap:
        (current_distance, current_node) = heapq.heappop(heap)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    
    return distances

def minimize_color_difference(colors, color_difference=delta_e_cie2000):
    colors_cielab = [convert_color(sRGBColor(*color), LabColor) for color in colors]
    graph = build_color_graph(colors_cielab, color_difference)
    start = 0
    min_k_dist = float('inf')
    for i, color in enumerate(colors):
        k_dist = color[0]**2 + color[1]**2 + color[2]**2
        if k_dist <  min_k_dist:
            min_k_dist = k_dist
            start = i
    new_graph = dijkstra(graph, start)
    sorted_graph = sorted(new_graph.items(), key = lambda item: item[1])
    return [colors[i[0]] for i in sorted_graph]


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    import random

    N = 40
    colors = [(random.random(), random.random(), random.random()) for i in range(N)]

    minimized_colors1 = minimize_color_difference(colors, delta_e_cie1976)
    minimized_colors2 = minimize_color_difference(colors, delta_e_cie1994)
    minimized_colors3 = minimize_color_difference(colors, delta_e_cie2000)
    minimized_colors4 = minimize_color_difference(colors, delta_e_cmc)

    x0 = [0 for i in range(N)]
    x1 = [1 for i in range(N)]
    x2 = [2 for i in range(N)]
    x3 = [3 for i in range(N)]
    x4 = [4 for i in range(N)]
    y = numpy.arange(N)

    plt.scatter(x0, y, color=colors)
    plt.scatter(x1, y, color=minimized_colors1)
    plt.scatter(x2, y, color=minimized_colors2)
    plt.scatter(x3, y, color=minimized_colors3)
    plt.scatter(x4, y, color=minimized_colors4)

    plt.show()

