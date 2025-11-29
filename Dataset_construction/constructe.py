import tool
import math
import networkx as nx
import random

def construct_matrix(nodes, faulty_nodes, res_count_PMC, res_count_MMX):
    matrix = []
    for node in nodes:
        pmc_value = res_count_PMC.get(node, 0)
        mmx_value = res_count_MMX.get(node, 0)
        fault_status = 'F' if node in faulty_nodes else 'FF'
        matrix.append([node, pmc_value, mmx_value, fault_status])

    return matrix


def construct_matrix_Power(G, nodes, faulty_nodes, res_count_PMC, res_count_MMX):
    matrix = []
    for node in nodes:
        pmc_value = res_count_PMC.get(node, 0)
        mmx_value = res_count_MMX.get(node, 0)
        degree_value = G.degree(node)
        fault_status = 'F' if node in faulty_nodes else 'FF'
        matrix.append([node, pmc_value, mmx_value, degree_value, fault_status])

    return matrix

def data_gen_Sn(n, num_faulty_nodes):
    G = tool.generate_graph_Sn(n)
    faulty_nodes = tool.generate_random_fault_nodes(G, num_faulty_nodes)
    PMC_res = tool.PMC(G, faulty_nodes)
    MMX_res = tool.MMX(G, faulty_nodes)
    res_count_PMC = tool.count_value_occurrences_PMC(PMC_res)
    res_count_MMX = tool.count_value_occurrences_MMX(MMX_res)
    nodes = G.nodes()
    res = construct_matrix(nodes, faulty_nodes, res_count_PMC, res_count_MMX)
    filename = f'S{n}_nodes{math.factorial(n)}_numFaulty{num_faulty_nodes}'
    tool.save_to_csv(filename, res)

def data_gen_Qn(n, num_faulty_nodes):
    G = tool.generate_graph_Qn(n)
    faulty_nodes = tool.generate_random_fault_nodes(G, num_faulty_nodes)
    PMC_res = tool.PMC(G, faulty_nodes)
    MMX_res = tool.MMX(G, faulty_nodes)
    res_count_PMC = tool.count_value_occurrences_PMC(PMC_res)
    res_count_MMX = tool.count_value_occurrences_MMX(MMX_res)
    nodes = G.nodes()
    res = construct_matrix(nodes, faulty_nodes, res_count_PMC, res_count_MMX)
    filename = f'Q{n}_nodes{2 ** n}_numFaulty{num_faulty_nodes}'
    tool.save_to_csv(filename, res)

def data_gen_BCubenk(n, k, num_faulty_nodes):
    G = tool.generate_graph_BCubenk(n, k)
    print(G)
    print(G.nodes())
    print(G.edges())
    faulty_nodes = tool.generate_random_fault_nodes(G, num_faulty_nodes)
    PMC_res = tool.PMC(G, faulty_nodes)
    MMX_res = tool.MMX(G, faulty_nodes)
    res_count_PMC = tool.count_value_occurrences_PMC(PMC_res)
    res_count_MMX = tool.count_value_occurrences_MMX(MMX_res)
    nodes = G.nodes()
    res = construct_matrix(nodes, faulty_nodes, res_count_PMC, res_count_MMX)
    filename = f'BCube({n},{k})_nodes{len(G.nodes())}_numFaulty{num_faulty_nodes}'
    tool.save_to_csv(filename, res)
