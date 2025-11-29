import numpy as np
import itertools
from itertools import permutations,combinations
import networkx as nx
import random
import csv
import datetime
from collections import Counter
import os
import pandas as pd
from packaging import version
import re
def generate_graph_Sn(n):
    numbers = list(range(1, n + 1))
    perms = list(permutations(numbers))
    nodes = [''.join(map(str, perm)) for perm in perms]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for perm in nodes:
        for i in range(1, len(perm)):
            str1 = list(perm)
            str1[0], str1[i] = str1[i], str1[0]
            edge = (perm, ''.join(str1))
            G.add_edge(*edge)
    return G





def generate_graph_Qn(n):
    def backtrack(curr_permutation):
        if len(curr_permutation) == n:
            vertices.append(curr_permutation)
            return
        backtrack(curr_permutation + '0')
        backtrack(curr_permutation + '1')
    vertices = []
    backtrack('')
    edges = []
    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if hamming_distance(vertices[i], vertices[j]) == 1:
                edges.append((vertices[i], vertices[j]))
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    return G

def generate_graph_BCubenk(n, k):
    G = nx.Graph()
    num_servers = n ** (k + 1)

    for i in range(num_servers):
        # 计算节点维度
        dimension = i // (n ** (k + 1))  # 根据节点索引计算维度
        G.add_node(f"S{i}", dimension=dimension)

    for level in range(k + 1):
        step = n ** level
        for i in range(0, num_servers, step * n):
            for j in range(step):
                for pair in combinations(range(n), 2):
                    server1 = i + j + pair[0] * step
                    server2 = i + j + pair[1] * step
                    G.add_edge(f"S{server1}", f"S{server2}")

    return G


def generate_random_fault_nodes(G, fault_nodes_num):
    if fault_nodes_num > G.number_of_nodes():
        raise ValueError("Fault nodes number cannot be greater than total nodes.")

    nodes = list(G.nodes())
    while True:
        random_fault_nodes = set(np.random.choice(nodes, size=fault_nodes_num, replace=False))
        return list(random_fault_nodes)


def PMC(G, faulty_nodes):
    if isinstance(faulty_nodes, np.ndarray):
        faulty_nodes = set(faulty_nodes)
    node_index = {node: idx for idx, node in enumerate(G)}
    test_results = {}

    for u in G:
        u_idx = node_index[u]
        test_results[u] = []

        for v in G[u]:
            v_idx = node_index[v]
            if G.has_edge(u, v):
                if u in faulty_nodes:
                    if random.choice([0, 1]) == 1:
                        test_results[u].append(v)
                else:
                    if v in faulty_nodes:
                        test_results[u].append(v)

    test_results = {k: v for k, v in test_results.items() if v}

    return test_results


def MMX(G, faulty_nodes):
    if isinstance(faulty_nodes, np.ndarray):
        faulty_nodes = set(faulty_nodes)
    node_index = {node: idx for idx, node in enumerate(G)}
    test_results_MMX = {}

    for node in G.nodes():
        test_results_MMX[node] = []
        neighbors = list(G.neighbors(node)) 
        if len(neighbors) % 2 != 0:
            random_neighbor = random.choice(neighbors)
            neighbors.append(random_neighbor)
        neighbor_pairs = [neighbors[i:i + 2] for i in range(0, len(neighbors), 2)]
        for neighbor_pair in neighbor_pairs:
            if node in faulty_nodes:
                if random.choice([0, 1]) == 1:
                    test_results_MMX[node].append(neighbor_pair)
            else:
                if neighbor_pair[0] in faulty_nodes or neighbor_pair[1] in faulty_nodes:
                    test_results_MMX[node].append(neighbor_pair)

    test_results_MMX = {k: v for k, v in test_results_MMX.items() if v}
    return test_results_MMX


def count_value_occurrences_PMC(data_dict):
    all_values = []
    for key, value_list in data_dict.items():
        all_values.extend(value_list)
    value_counts = Counter(all_values)

    return dict(value_counts)


def count_value_occurrences_MMX(data_dict):
    all_values = []
    for key, value_list in data_dict.items():
        for sublist in value_list:
            all_values.extend(sublist) 
    value_counts = Counter(all_values)

    return dict(value_counts)

def save_to_csv(filename,matrix):
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename_ = f"{current_time}_{filename}.csv"
    headers = ['node', 'sum_PMC', 'sum_MM*','label']
    with open(filename_, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(matrix)

    print(f"Matrix saved to {filename_}")

def save_to_csv_Power(filename,matrix):
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename_ = f"{current_time}_{filename}.csv"
    headers = ['node', 'sum_PMC', 'sum_MM*','degree','label']
    with open(filename_, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(matrix) 

    print(f"Matrix saved to {filename_}")


def calculate_degree(G):
    return dict(G.degree())

def merge_csv_files(input_dir='.', output_file='merged.csv'):
    csv_files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.csv') and f != output_file
    ]

    if not csv_files:
        print(f"在目录 {input_dir} 中未找到CSV文件")
        return

    csv_files.sort()
    dfs = []
    columns = None

    for i, filename in enumerate(csv_files):
        file_path = os.path.join(input_dir, filename)
        try:
            if i == 0:
                df = pd.read_csv(file_path, dtype=str)
                columns = df.columns.tolist()
                dfs.append(df)
            else:
                df = pd.read_csv(
                    file_path,
                    header=None,
                    skiprows=1,
                    names=columns,
                    dtype=str
                )
                dfs.append(df)
        except (pd.errors.ParserError, ValueError) as e:
            print(f"文件 {filename} 格式错误或列不匹配，已跳过。错误：{str(e)}")
            continue

    if not dfs:
        print("无有效文件可合并")
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(input_dir, output_file)
    merged_df.to_csv(output_path, index=False)
    print(f"合并完成！共合并 {len(csv_files)} 个文件，保存至：{output_path}")


def insert_column_to_csv(input_path, insert_position, new_header, new_value, output_dir=None):
    if os.path.isdir(input_path):
        process_directory(input_path, insert_position, new_header, new_value, output_dir)
        return

    df = pd.read_csv(input_path, dtype=str)
    insert_idx = max(0, insert_position - 1)
    df.insert(insert_idx, new_header, new_value)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(input_path))
    else:
        output_path = input_path

    df.to_csv(output_path, index=False)
    print(f"处理完成：{input_path} -> {output_path}")
def process_directory(input_dir, insert_pos, header, value, output_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"目录不存在: {input_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            insert_column_to_csv(file_path, insert_pos, header, value, output_dir)


def shuffle_csv_rows(input_path, output_path, header=True, encoding='utf-8'):
    with open(input_path, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        rows = list(reader)
    header_row = []
    if header and len(rows) > 0:
        header_row = [rows[0]]
        data_rows = rows[1:]
    else:
        data_rows = rows
    random.shuffle(data_rows)
    with open(output_path, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        if header_row:
            writer.writerows(header_row)
        writer.writerows(data_rows)


def calculate_single_node_betweenness(G, v):
    nodes = list(G.nodes)
    n = len(nodes)
    if n <= 1:
        return 0.0
    filtered_nodes = [node for node in nodes if node != v]
    num_filtered = len(filtered_nodes)
    if num_filtered < 2:
        return 0.0
    norm_factor = 2.0 / ((n - 1) * (n - 2)) if n > 2 else 0.0

    betweenness = 0.0

    # 遍历所有s < t的节点对，其中s和t都不等于v
    for i in range(num_filtered):
        s = filtered_nodes[i]
        for j in range(i + 1, num_filtered):
            t = filtered_nodes[j]
            # 检查s和t之间是否存在路径
            if not nx.has_path(G, s, t):
                continue
            try:
                # 获取所有最短路径
                all_paths = nx.all_shortest_paths(G, s, t)
            except nx.NetworkXNoPath:
                continue

            total_paths = 0
            count_v = 0
            for path in all_paths:
                total_paths += 1
                # 检查v是否在路径的中间部分（排除起点和终点）
                if v in path[1:-1]:
                    count_v += 1

            if total_paths > 0:
                betweenness += (count_v / total_paths)
    betweenness *= norm_factor

    return betweenness


def add_features(input_path, output_path, replace_zero_node=True):
    df = pd.read_csv(input_path, dtype={'node': str})
    if replace_zero_node:
        zero_pattern = r'^0+$'
        df['node'] = df['node'].replace(zero_pattern, np.nan, regex=True)
    df['ratio_PMC'] = df['sum_PMC'] / df['degree']
    df['ratio_MM*'] = df['sum_MM*'] / df['degree']
    df['diff_PMC'] = df['degree'] - df['sum_PMC']
    df['diff_MM*'] = df['degree'] - df['sum_MM*']

    cols = list(df.columns)
    insert_index = cols.index('degree') + 1
    new_cols = cols[:insert_index] + ['ratio_PMC', 'ratio_MM*', 'diff_PMC', 'diff_MM*'] + cols[insert_index:-4]
    df = df[new_cols]
    df.to_csv(output_path, index=False, float_format='%.4f')


def calculate_class_distribution(csv_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path, dtype={'node': str})

        # 检查标签列是否存在
        if 'label' not in df.columns:
            raise ValueError("CSV文件中缺少'label'列")

        # 统计类别分布
        class_counts = df['label'].value_counts(dropna=True)
        total_samples = len(df)

        # 计算百分比
        distribution = (class_counts / total_samples * 100).round(2).to_dict()

        return distribution

    except FileNotFoundError:
        print(f"错误：文件 {csv_path} 未找到")
        return {}
    except pd.errors.EmptyDataError:
        print("错误：文件内容为空")
        return {}
    except Exception as e:
        print(f"未知错误：{str(e)}")
        return {}


def merge_csv_with_metadata(input_dir, output_file):
    pattern = re.compile(r'^(.+?)_nodes(\d+)_numFaulty(\d+).*\.csv$')

    # 存储处理结果的列表
    dfs = []

    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        # 匹配文件名模式
        match = pattern.match(filename)
        if not match:
            continue

        # 从文件名解析元数据（转换为整数）
        graph_name = match.group(1)  # 图名
        total_nodes = int(match.group(2))  # 总点数（转换为整数）
        faulty_nodes = int(match.group(3))  # 故障点数（转换为整数）

        # 构建完整文件路径
        filepath = os.path.join(input_dir, filename)

        try:
            # 读取CSV文件内容（不包含表头）
            df = pd.read_csv(filepath, skiprows=1, header=None)

            # 添加元数据列
            df.insert(0, '故障点数', faulty_nodes)
            df.insert(0, '总点数', total_nodes)
            df.insert(0, '图名', graph_name)

            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"警告：文件 {filename} 内容为空，已跳过")
            continue

    if not dfs:
        print("错误：没有找到有效文件")
        return

    # 合并所有数据
    merged_df = pd.concat(dfs, ignore_index=True)

    # 获取原始列名（使用第一个文件的表头）
    with open(os.path.join(input_dir, os.listdir(input_dir)[0])) as f:
        original_header = f.readline().strip().split(',')

    # 构建完整列名
    final_columns = ['图名', '总点数', '故障点数'] + original_header

    # 添加列名并排序
    merged_df.columns = final_columns
    merged_df = merged_df.sort_values(by=['图名', '故障点数'], ascending=[True, True])
    merged_df.reset_index(drop=True, inplace=True)

    # 保存合并结果
    merged_df.to_csv(output_file, index=False)
    print(f"成功合并 {len(dfs)} 个文件到 {output_file}")


def load_power_dataset(file_path):
    """Load Power dataset from edge list file"""
    edges = []
    node_set = set()

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    edges.append((u, v))
                    node_set.add(u)
                    node_set.add(v)

    # Create graph
    G = nx.Graph()
    G.add_edges_from(edges)

    print(f"Graph info: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Node range: {min(node_set)} to {max(node_set)}")

    return G

#函数：将输入的图G转换为邻接矩阵
def graph_to_adj_matrix(G):
    return nx.to_numpy_array(G)

#函数：获取输入图G的点数
def get_num_nodes(G):
    return G.number_of_nodes()

#函数：获取输入图G的边数
def get_num_edges(G):
    return G.number_of_edges()

#函数：获取输入图G的点集
def get_nodes(G):
    return list(G.nodes())

#函数：获取输入图G的边集
def get_edges(G):
    return list(G.edges())

#函数：获取输入图G中k条不相邻的边
def get_disjoint_edges(G, k):
    edges = list(G.edges())
    # 按顶点度排序，优先选择连接低度顶点的边

    selected_edges = []
    used_vertices = set()

    for edge in edges:
        if len(selected_edges) >= k:
            break

        u, v = edge
        if u not in used_vertices and v not in used_vertices:
            selected_edges.append(edge)
            used_vertices.add(u)
            used_vertices.add(v)

    return selected_edges

#函数：获取输入边集edges的全部邻居
def get_all_neighbors(Graph, edges):
    neighbor = set()
    for e in edges:
        node1, node2 =  e
        print(node1, node2)
        for d in list(nx.neighbors(Graph, node1)):
            neighbor.add(d)
        for d in list(nx.neighbors(Graph, node2)):
            neighbor.add(d)
    return set(neighbor)
