from typing import List, Dict, Tuple

import networkx as nx


def count_word_fre(candidates: List[Tuple[str, Tuple[int, int]]]) -> Dict[str, int]:
    cans: Dict[str, int] = {}
    for v in candidates:
        if v[0] not in cans.keys():
            cans[v[0]] = 0
        cans[v[0]] += 1
    return cans


def candidates_to_graph(candidates: List[Tuple[str, Tuple[int, int]]]) -> nx.Graph:
    G = nx.Graph()
    word_freq = count_word_fre(candidates=candidates)  # 词频表

    # 向图中添加点
    for word, freq in word_freq.items():
        G.add_node(word, frequency=freq)

    for i in range(len(candidates)):
        # 最后一个候选词，只统计词频
        if i == len(candidates) - 1:
            continue
        a_word = candidates[i][0]
        b_word = candidates[i + 1][0]
        a_pos = candidates[i][1][1] - 1  # 候选词a的结束位置
        b_pos = candidates[i + 1][1][0]  # 候选词b的开始位置
        # 得到a和b之间的边权重
        if a_pos + 2 >= b_pos:
            row_ab = 1
        else:
            row_ab = 0
        if row_ab == 1:
            if not G.has_edge(a_word, b_word):
                G.add_edge(a_word, b_word)
                G[a_word][b_word]['weight'] = 0
            G[a_word][b_word]['weight'] += row_ab

    # 计算节点的pagerank权重
    pagerank_res = run_pagerank(G)
    nodes = pagerank_res.keys()

    # 修改节点权重属性 = 词频 * pagerank权重
    for node in nodes:
        G.nodes[node]['weight'] = G.nodes[node]['frequency'] * pagerank_res[node]

    # 修改边权重 = 词共现次数 * 0.5 * （两节点的权重之和）
    for i in range(len(word_freq) - 1):
        for j in range(i + 1, len(word_freq)):
            words = [key for key, value in word_freq.items()]
            a_word = words[i]
            b_word = words[j]
            if G.has_edge(a_word, b_word):
                G[a_word][b_word]['weight'] = G[a_word][b_word]['weight'] * 0.5 * (
                        G.nodes[a_word]['weight'] + G.nodes[b_word]['weight'])

    return G


def run_pagerank(graph: nx.Graph, personalization=None) -> dict:
    if personalization:
        return nx.pagerank(
            G=graph, alpha=0.85, tol=0.0001, weight='weight', personalization=personalization)
    else:
        return nx.algorithms.link_analysis.pagerank(G=graph, alpha=0.85, tol=0.0001, weight='weight')


def buildgraph(graph: nx.Graph):
    pass
