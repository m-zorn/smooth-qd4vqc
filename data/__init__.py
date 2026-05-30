import networkx as nx
import numpy as np
from typing import List, Tuple
import cvxpy as cp
from scipy.linalg import sqrtm
from tqdm import trange
from src.utils import to_z_basis
import random

def goemans_williamson_solution(graph:List[Tuple[int, int]], vertices:int):
    ''' The approximate solution using the Goemans-Williamson Algorithm. Has aproximation-ratio of 0.87.
    For explanation see https://www.youtube.com/watch?v=aFVnWq3RHYU '''
    X = cp.Variable((vertices, vertices), symmetric=True)
    constraints = [X >> 0]
    constraints += [
         X[i, i] == 1 for i in range(vertices)
    ]
    objective = sum(0.5*(1 - X[i,j]) for i, j in graph)
    prob = cp.Problem(cp.Maximize(objective), constraints).solve() #pos. semidefinite matrix
    x = sqrtm(X.value)
    u = np.random.randn(vertices) #normal to random hyperplane
    x = np.sign(x @ u) #rounding operation
    return x

def brute_force(nx_graph, hamiltonian):
    vertices = len(nx_graph.nodes)
    # Check for weights to pass (u,v,w) tuples if relevant
    # Note: nx.is_weighted(g) checks if ALL edges have weight, or ANY? 
    # It checks if ANY edge has 'weight' attribute.
    # We want consistent behavior.
    if nx.is_weighted(nx_graph):
        edges = list(nx_graph.edges(data='weight', default=1.0))
    else:
        edges = list(nx_graph.edges)
        
    possible_solutions = [format(i, f'0{vertices}b') for i in range(2**vertices)] #assignment solutions in binary encoding
    solutions = [hamiltonian(to_z_basis(possible_solution), edges) for possible_solution in possible_solutions]
    if max(solutions)==0:
        print("best solution is zero")
    return {
        "best_energy":max(solutions),
        "worst_energy":min(solutions),
        "best_solutions_total":solutions.count(max(solutions)),
        "solutions_total":len(solutions),
        "best_solution_ratio":solutions.count(max(solutions))/len(solutions),
        "num_unique_solutions":len(np.unique(solutions)),
        #"gw_solution":goemans_williamson_solution(edges, vertices)
    }

# The four test graphs from the QNEAT paper 
random = {"name":"random" ,"graph": [(0,1),(0,6),(0,7),(1,2),(1,6),(2,4),(2,7),(3,7),(3,6),(4,5),(4,6),(1,5),(5,6),(5,7),(6,7)], "n_nodes":8, "solution":{'best_energy': 12.0, 'best_solutions_total': 6, 'solutions_total': 256, 'best_solution_ratio': 0.0234375,'num_unique_solutions': 12}, 'edgeDensity':0.536}
ladder = {"name": "ladder", "graph": [(0,1),(0,7),(1,2),(2,3),(2,7),(3,4),(3,6),(4,5),(5,6),(6,7)], "n_nodes":8, "solution":{'best_energy': 10.0, 'best_solutions_total': 2, 'solutions_total': 256, 'best_solution_ratio': 0.0078125, 'num_unique_solutions': 9}, 'edgeDensity':0.357}
barbell = {"name": "barbell", "graph": [(0,1),(0,2),(0,3),(1,2),(1,3),(2,4),(2,3),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)], "n_nodes":8, "solution":{'best_energy': 9.0, 'best_solutions_total': 18, 'solutions_total': 256, 'best_solution_ratio': 0.0703125, 'num_unique_solutions': 9}, 'edgeDensity':0.464}
caveman = {"name": "caveman", "graph": [(0,1),(0,2),(1,2),(1,3),(1,4),(2,3),(3,5),(4,5),(4,6),(5,6),(5,7),(6,7)], "n_nodes":8, "solution":{'best_energy': 10.0, 'best_solutions_total': 2, 'solutions_total': 256, 'best_solution_ratio': 0.0078125, 'num_unique_solutions': 10}, 'edgeDensity':0.429}

def new_erdos_renyi_graph(num_vertices:int, connectivity:float=-1, seed:int=0):
    if connectivity == -1:
            connectivity = np.random.sample() + 1e-10
    assert 0 < connectivity <= 1
    return nx.erdos_renyi_graph(num_vertices, connectivity, seed=seed), connectivity

def new_weighted_erdos_renyi_graph(num_vertices:int, connectivity:float=-1, seed:int=0):
    g, p = new_erdos_renyi_graph(num_vertices, connectivity, seed)
    # Add random weights
    rng = np.random.default_rng(seed)
    for u, v in g.edges():
        g[u][v]['weight'] = round(rng.uniform(0.1, 1.0), 2)
    return g, p

def new_problem_set(num_graphs:int, num_vertices:int, cost_hamiltonian, connectivity:float=-1, seed:int=0, pbar:bool=False):
    problem_data = []
    name=f"erdos_reny_v{num_vertices}"
    if pbar:
        iterator = trange(num_graphs, desc="generating problem set") 
    else:
        iterator = range(num_graphs)
    for i in iterator:
        g, p = new_erdos_renyi_graph(num_vertices, connectivity, seed=seed+i)
        j = 1
        while not nx.is_connected(g):
            g, p = new_erdos_renyi_graph(num_vertices, connectivity, seed=seed+i+j)
            j += 1
        solution_dict = brute_force(g, cost_hamiltonian)
        problem_data.append({"name":name ,"graph": list(g.edges) , "n_nodes":num_vertices, "solution":solution_dict, 'edgeDensity':nx.density(g)})
        
    return problem_data

def new_weighted_problem_set(num_graphs:int, num_vertices:int, cost_hamiltonian, connectivity:float=-1, seed:int=0, pbar:bool=False):
    problem_data = []
    name=f"weighted_erdos_reny_v{num_vertices}"
    if pbar:
        iterator = trange(num_graphs, desc="generating weighted problem set")
    else: 
        iterator = range(num_graphs)
        
    for i in iterator:
        g, p = new_weighted_erdos_renyi_graph(num_vertices, connectivity, seed=seed+i)
        j = 1
        while not nx.is_connected(g):
            g, p = new_weighted_erdos_renyi_graph(num_vertices, connectivity, seed=seed+i+j)
            j += 1
        
        # brute_force handles weighting via nx.is_weighted(g) check
        solution_dict = brute_force(g, cost_hamiltonian)
        
        # Store edges as (u, v, w) tuples
        edges_with_weights = list(g.edges(data='weight', default=1.0))
        
        problem_data.append({
            "name":name,
            "graph": edges_with_weights, 
            "n_nodes":num_vertices, 
            "solution":solution_dict, 
            'edgeDensity':nx.density(g)
        })
        
    return problem_data
        

def to_z_basis(binary_solution:str):
    return ["-1" if c == "1" else "1" for c in list(binary_solution)]