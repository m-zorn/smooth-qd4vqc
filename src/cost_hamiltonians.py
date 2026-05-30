from src.utils import nx_graph_from

def clean_edge(e):
    # Only unpack first two elements if edges are weighted (u, v, w)
    return e[:2] if len(e) >= 3 else e

def max_cut_hamiltonian(assignment, graph_edges):
    score = 0
    for edge in graph_edges:
        if len(edge) == 3:
            u, v, w = edge
        else:
            u, v = edge
            w = 1.0
        # Metric: w * 0.5 * (z_i * z_j - 1). 
        # If z_i != z_j -> -1 * -1 = +1 (cut) -> +w
        # If z_i == z_j -> 0 (no cut)
        score += w * 0.5 * (int(assignment[u]) * int(assignment[v]) - 1)
    return -score

def weighted_max_cut_hamiltonian(assignment, graph_edges):
    return max_cut_hamiltonian(assignment, graph_edges)

def min_vertex_cover_hamiltonian(assignment, graph_edges):
    edge_penalty = sum([int(assignment[z_i]) * int(assignment[z_j]) + int(assignment[z_i]) + int(assignment[z_j]) for z_i, z_j in [clean_edge(e) for e in graph_edges]])
    vertex_penality = sum([int(i) for i in assignment] )
    return -(3*edge_penalty - vertex_penality)

def max_independed_set_hamiltonian(assignment, graph_edges):
    edge_penalty = sum([int(assignment[z_i]) * int(assignment[z_j]) - int(assignment[z_i]) - int(assignment[z_j]) for z_i, z_j in [clean_edge(e) for e in graph_edges]])
    vertex_penalty = sum([int(i) for i in assignment] )
    return -(3*edge_penalty + vertex_penalty)

def max_clique_hamiltonian(assignment, graph_edges):
    # Only unpack first two elements if edges are weighted (u, v, w)
    # nx_graph_from handles (u,v,w) -> (u,v) via add_weighted_edges_from and complement logic
    graph_edges_complement = nx_graph_from(len(assignment), graph_edges, complement=True).edges
    edge_penalty = sum([int(assignment[z_i]) * int(assignment[z_j]) - int(assignment[z_i]) - int(assignment[z_j]) for z_i, z_j in graph_edges_complement])
    vertex_penalty = sum([int(i) for i in assignment] )
    return -(3*edge_penalty + vertex_penalty)

