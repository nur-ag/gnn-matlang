import igraph as ig
import networkx as nx
from collections import defaultdict, Counter

import sys
sys.path.append('../IGEL/src')
sys.path.append('../IGEL/gnnml-comparisons')
from structural import StructuralMapper

SR25_PATH = "dataset/sr25/raw/sr251256.g6"
GRAPH8C_PATH = "dataset/graph8c/raw/graph8c.g6"


def nx_to_ig(nx_G):
    G = ig.Graph()
    G.add_vertices(list(nx_G.nodes))
    G.add_edges(list(nx_G.edges))
    return G


def find_isomorphisms(Gs, encs):
    mapping = defaultdict(list)
    for G, enc in zip(Gs, encs):
        mapping[enc].append(G)
    non_isomorphic = []
    for enc, graphs in mapping.items():
        for i, G_a in enumerate(graphs):
            for j, G_b in enumerate(graphs[i+1:], i+1):
                if not G_a.isomorphic(G_b):
                    result_tuple = (enc, i, j)
                    non_isomorphic.append(result_tuple)
    return non_isomorphic


# Load the input dataset -- SR25 or Graph8c
# As long as the input is graph6-formatted, we can check distinguishability anywhere
input_param = sys.argv[1] if len(sys.argv) >= 2 else "graph8c"
if input_param.lower() == "sr25":
    input_path = SR25_PATH
elif input_param.lower() == "graph8c":
    input_path = GRAPH8C_PATH
else:
    input_path = input_param

# Load graphs and convert to igraph
input_gs = nx.read_graph6(input_path)
gs = [nx_to_ig(G) for G in input_gs]


# Compute all IGEL encodings at a max. encoding diameter
# And also the 'total' IGEL encodings subsuming all values of alpha
total_encs = []
max_diam = int(sys.argv[2]) if len(sys.argv) >= 3 else max([G.diameter() for G in gs])
encs = [[] for _ in range(max_diam)]
for G in gs:
    total_enc = []
    for distance in range(1, max_diam + 1):
        igel_enc = StructuralMapper(G, distance=distance, use_distances=True, cache_field=f'igel_d{distance}', num_workers=1)
        igel_result = Counter([tuple(sorted(zip(*x))) for x in igel_enc.mapping(G.vs, G)]).most_common()
        result_tuple = (distance, tuple(igel_result))
        total_enc.append(result_tuple)
        encs[distance - 1].append(result_tuple)
    total_enc = tuple(total_enc)
    total_encs.append(total_enc)

verbose = sys.argv[3].lower() == 'print' if len(sys.argv) >= 3 else False
if verbose:
    print(f"The max. diameter is {max_diam}.")
    print(f"The encodings are: {[set(enc) for enc in encs]}")
    print(f"The aggregate encodings are: {set(total_encs)}")

# Identify distinguishability failures
per_distance_fails = [(distance, f) for distance in range(1, max_diam + 1)
                                    for f in find_isomorphisms(gs, encs[distance - 1])]
total_distance_fails = find_isomorphisms(gs, total_encs)

# List the value of alpha for failed isomorphisms
print(per_distance_fails)
print("Undistinguishability frequency by α: ", Counter([distance for distance, _ in per_distance_fails]).most_common())
print("Undistinguishability frequency by all possible α combined: ", Counter([encoding[0] for encoding, _, _ in total_distance_fails]).most_common())
