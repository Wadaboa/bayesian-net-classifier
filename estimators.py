'''
Custom BN structure estimators
'''


import itertools

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import StructureEstimator
from pgmpy.base import DAG
from sklearn.metrics import mutual_info_score


def conditional_mutual_info_score(xi, xj, c):
    '''
    Compute conditional mutual information I(Xi, Xj | C), given
    numpy arrays or pandas series for xi and xj and a numpy matrix
    or a a pandas dataframe for c.
    In this implementation c can be given as a single column or as
    multiple columns.
    '''
    conditions = pd.DataFrame(c)
    if len(conditions.columns) == 0:
        return mutual_info_score(xi, xj)
    cond_mutual_info = 0
    for _, cond in conditions.iteritems():
        unique_condition_values = cond.unique()
        for i in unique_condition_values:
            condition_proba = np.sum(cond == i) / len(cond)
            cond_mutual_info += mutual_info_score(
                xi[cond == i], xj[cond == i],
            ) * condition_proba
    return np.sum(cond_mutual_info)


def simple_conditional_mutual_info_score(df, xi, xj, c):
    '''
    Compute conditional mutual information I(Xi, Xj | C), given
    a pandas dataframe and column names for xi, xj and c.
    In this implementation c can only be given as a single column.
    '''
    unique_xi_values = df[xi].unique()
    unique_xj_values = df[xj].unique()
    unique_c_values = df[c].unique()
    scores = []
    for i, j, k in itertools.product(unique_xi_values, unique_xj_values, unique_c_values):
        prob_ijk = len(
            df[(df[xi] == i) & (df[xj] == j) & (df[c] == k)]
        ) / len(df)
        rdf = df[df[c] == k]
        prob_ij_k = len(rdf[(rdf[xi] == i) & (rdf[xj] == j)]) / len(rdf)
        prob_i_k = len(rdf[rdf[xi] == i]) / len(rdf)
        prob_j_k = len(rdf[rdf[xj] == j]) / len(rdf)
        current_score = prob_ijk * np.log(
            prob_ij_k / (prob_i_k * prob_j_k + 10e-5)
        )
        scores.append(current_score)
    return np.nansum(scores)


class TreeAugmentedNaiveBayesSearch(StructureEstimator):

    def __init__(self, data, class_node, root_node=None, **kwargs):
        '''
        Search class for learning tree-augmented naive bayes (TAN) graph structure with a given set of variables.
        TAN is an extension of Naive Bayes classifer and allows a tree structure over the independent variables
        to account for interaction.
        See https://github.com/pgmpy/pgmpy/pull/1266/commits for reference.
        '''
        self.class_node = class_node
        self.root_node = root_node

        super().__init__(data, **kwargs)

    def estimate(self):
        '''
        Estimates the DAG structure that fits best to the given data set using the Chow-Liu algorithm.
        Only estimates network structure, no parametrization.
        '''
        if self.class_node not in self.data.columns:
            raise ValueError("Class node must exist in data")

        if self.root_node is not None and self.root_node not in self.data.columns:
            raise ValueError("Root node must exist in data")

        graph = nx.Graph()
        df_features = self.data.loc[:, self.data.columns != self.class_node]
        total_cols = len(df_features.columns)
        for i in range(total_cols):
            from_node = df_features.columns[i]
            graph.add_node(from_node)
            for j in range(i + 1, total_cols):
                to_node = df_features.columns[j]
                graph.add_node(to_node)
                mi = mutual_info_score(
                    df_features.iloc[:, i], df_features.iloc[:, j]
                )
                graph.add_edge(from_node, to_node, weight=mi)
        tree = nx.maximum_spanning_tree(graph)

        if self.root_node:
            digraph = nx.bfs_tree(tree, self.root_node)
        else:
            digraph = nx.bfs_tree(tree, df_features.columns[0])

        for node in df_features.columns:
            digraph.add_edge(self.class_node, node)

        return DAG(digraph)


class BNAugmentedNaiveBayesSearch(StructureEstimator):

    def __init__(self, data, class_node, epsilon=0.003, **kwargs):
        '''
        Search class for learning BN-augmented naive Bayes (BAN) graph structure with a given set of variables.
        BAN is an extension of Naive Bayes classifer which allows a graph structure over the independent variables
        to account for interaction.
        See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.3241&rep=rep1&type=pdf for reference.
        '''
        self.class_node = class_node
        self.epsilon = epsilon

        super().__init__(data, **kwargs)

    def estimate(self):
        '''
        Estimates the DAG structure that fits best to the given data set using the CBL2 algorithm.
        Only estimates network structure, no parametrization.
        '''
        if self.class_node not in self.data.columns:
            raise ValueError("Class node must exist in data")

        ################## Drafting #####################

        L = []
        graph = nx.Graph()
        df_features = self.data.loc[
            :, self.data.columns != self.class_node
        ]
        total_cols = len(df_features.columns)
        graph.add_nodes_from(df_features.columns)
        for i in range(total_cols):
            from_node = df_features.columns[i]
            for j in range(i + 1, total_cols):
                to_node = df_features.columns[j]
                mi = conditional_mutual_info_score(
                    df_features.iloc[:, i], df_features.iloc[:, j],
                    self.data.loc[:, self.class_node]
                )
                if mi > self.epsilon:
                    L.append((from_node, to_node, mi))

        # Sort pairs of nodes by decreasing mutual information
        L.sort(key=lambda tup: tup[2], reverse=True)

        # Get the first two pairs of nodes and add corresponding edges
        from_node, to_node, mi = L.pop(0)
        graph.add_edge(from_node, to_node, weight=mi)
        from_node, to_node, mi = L.pop(0)
        graph.add_edge(from_node, to_node, weight=mi)

        # If there is no adjacency path between a pair of nodes, add the corresponding edge
        lenght = len(L)
        i = 0
        while i < lenght:
            from_node, to_node, mi = L[i]
            if len(graph.edges) - 1 == len(graph.nodes):
                break
            if not nx.has_path(graph, from_node, to_node):
                graph.add_edge(from_node, to_node, weight=mi)
                L.pop(i)
                lenght -= 1
                i -= 1
            i += 1

        ################## Thickening ###################

        for i in range(len(L)):
            from_node, to_node, mi = L[i]
            if not self.try_to_separate_a(graph, from_node, to_node):
                graph.add_edge(from_node, to_node, weight=mi)

        ################## Thinning #####################

        edges = list(graph.edges)
        for edge in edges:
            from_node, to_node = edge
            graph.remove_edge(from_node, to_node)
            if (nx.has_path(graph, from_node, to_node) and
                    not self.try_to_separate_a(graph, from_node, to_node)):
                graph.add_edge(from_node, to_node)

        edges = list(graph.edges)
        for edge in edges:
            from_node, to_node = edge
            graph.remove_edge(from_node, to_node)
            if (nx.has_path(graph, from_node, to_node) and
                    not self.try_to_separate_b(graph, from_node, to_node)):
                graph.add_edge(from_node, to_node)

        # ORIENT EDGES DOES NOT WORK
        # oriented_edges = self.orient_edges(graph)
        # digraph = nx.DiGraph(oriented_edges)
        digraph = nx.dfs_tree(graph, df_features.columns[0])

        for node in df_features.columns:
            digraph.add_edge(self.class_node, node)

        return DAG(digraph)

    def try_to_separate_a(self, graph, node1, node2):
        node1_neighbors = set(nx.neighbors(graph, node1))
        node2_neighbors = set(nx.neighbors(graph, node2))
        n1 = set()
        n2 = set()
        for path in nx.all_simple_paths(graph, source=node1, target=node2):
            for node in path[1:-1]:
                if node in node1_neighbors:
                    n1.add(node)
                if node in node2_neighbors:
                    n2.add(node)

        # Remove the currently known child-nodes of node1 from N1
        # and child-nodes of node2 from N2 (?)

        if len(n1) > len(n2):
            n1, n2 = n2, n1

        skip_step = False
        n2_used = False
        c = set(n1)
        while True:
            if not skip_step:
                v = conditional_mutual_info_score(
                    self.data[node1], self.data[node2],
                    self.data[c.union({self.class_node})]
                )
                if v < self.epsilon:
                    return True

            if len(c) == 1:
                if not n2_used:
                    c = set(n2)
                    n2_used = True
                else:
                    return False
            else:
                values = []
                conditions = []
                for node in graph.nodes:
                    ci = {n for n in c if n != node}
                    conditions.append(ci)
                    vi = conditional_mutual_info_score(
                        self.data[node1], self.data[node2],
                        self.data[ci.union({self.class_node})]
                    )
                    values.append(vi)
                min_index = np.argmin(values)
                vm = values[min_index]
                cm = conditions[min_index]
                if vm < self.epsilon:
                    return True
                elif vm > v:
                    if not n2_used:
                        c = set(n2)
                        n2_used = True
                    else:
                        return False
                else:
                    v = vm
                    c = cm
                    skip_step = True
                    continue

            skip_step = False

        return False

    def try_to_separate_b(self, graph, node1, node2):
        node1_neighbors = set(nx.neighbors(graph, node1))
        node2_neighbors = set(nx.neighbors(graph, node2))
        n1 = set()
        n2 = set()
        paths = list(nx.all_simple_paths(graph, source=node1, target=node2))
        for path in paths:
            for node in path[1:-1]:
                if node in node1_neighbors:
                    n1.add(node)
                if node in node2_neighbors:
                    n2.add(node)

        n1_prime = set()
        n1_neighbors = set()
        for n in n1:
            n1_neighbors.update(set(nx.neighbors(graph, n)))
        for path in paths:
            for node in path[1:-1]:
                if node in n1_neighbors and node not in n1:
                    n1_prime.add(node)

        n2_prime = set()
        n2_neighbors = set()
        for n in n2:
            n2_neighbors.update(set(nx.neighbors(graph, n)))
        for path in paths:
            for node in path[1:-1]:
                if node in n2_neighbors and node not in n2:
                    n2_prime.add(node)

        if len(n1) + len(n1_prime) < len(n2) + len(n2_prime):
            c = set(n1)
            c.update(set(n1_prime))
        else:
            c = set(n2)
            c.update(set(n2_prime))

        while True:
            v = conditional_mutual_info_score(
                self.data[node1], self.data[node2],
                self.data[c.union({self.class_node})]
            )
            if v < self.epsilon:
                return True

            c_prime = set(c)
            for i in c:
                ci = {n for n in c if n != i}
                vi = conditional_mutual_info_score(
                    self.data[node1], self.data[node2],
                    self.data[ci.union({self.class_node})]
                )
                if vi < self.epsilon:
                    return True
                elif vi <= v + self.epsilon and i in c_prime:
                    c_prime.discard(i)

            if len(c_prime) < len(c):
                c = set(c_prime)
            else:
                return False

        return False

    def orient_edges(self, graph):
        oriented_edges = set()
        for s1, s2 in itertools.product(graph.nodes, repeat=2):
            new_oriented_edges = self.orient_edge(graph, s1, s2)
            for a, b in new_oriented_edges:
                if (b, a) not in oriented_edges:
                    oriented_edges.add((a, b))

        oriented_edges_list = list(oriented_edges)
        lenght = len(oriented_edges_list)
        for i in range(lenght):
            a, b = oriented_edges_list[i]
            for c in graph.nodes:
                c_neighbors = list(nx.neighbors(graph, c))
                if (b in c_neighbors and a not in c_neighbors and
                        (b, c) not in oriented_edges and (c, b) not in oriented_edges):
                    oriented_edges.add((b, c))

        not_oriented_edges = set(graph.edges).difference(oriented_edges)
        oriented_edges = list(oriented_edges)
        tmp_graph = nx.DiGraph(oriented_edges)
        for edge in not_oriented_edges:
            a, b = edge
            if (a in tmp_graph.nodes and b in tmp_graph.nodes and nx.has_path(tmp_graph, a, b) and
                    (b, a) not in oriented_edges):
                oriented_edges.append((a, b))

        return oriented_edges

    def orient_edge(self, graph, s1, s2):
        oriented_edges = []
        n1 = []
        n2 = []
        s1_neighbors = list(nx.neighbors(graph, s1))
        s2_neighbors = list(nx.neighbors(graph, s2))
        inter = set(s1_neighbors).intersection(set(s2_neighbors))
        paths = list(nx.all_simple_paths(graph, source=s1, target=s2))
        if (s1, s2) not in graph.edges and len(inter) > 0:
            for path in paths:
                for node in path[1:-1]:
                    if node in s1_neighbors:
                        n1.append(node)
                    if node in s2_neighbors:
                        n2.append(node)

        n1_prime = []
        n2_prime = []
        n1_neighbors = [list(nx.neighbors(graph, n)) for n in n1]
        n2_neighbors = [list(nx.neighbors(graph, n)) for n in n2]
        for path in paths:
            for node in path[1:-1]:
                if node not in n1 and node in n1_neighbors:
                    n1_prime.append(node)
                if node not in n2 and node in n2_neighbors:
                    n2_prime.append(node)

        if len(n1) + len(n1_prime) < len(n2) + len(n2_prime):
            c = list(n1)
            c.extend(list(n1_prime))
        else:
            c = list(n2)
            c.extend(list(n2_prime))

        while True:
            v = conditional_mutual_info_score(
                self.data[s1], self.data[s2],
                self.data[c + [self.class_node]]
            )
            if v < self.epsilon:
                return oriented_edges

            c_prime = list(c)
            for i in c:
                ci = [n for n in c if n != i]
                vi = conditional_mutual_info_score(
                    self.data[s1], self.data[s2],
                    self.data[ci + [self.class_node]]
                )
                if vi <= v + self.epsilon:
                    c_prime = [n for n in c_prime if n != i]
                    oriented_edges.append((s1, i))
                    oriented_edges.append((s2, i))
                elif vi <= self.epsilon:
                    return oriented_edges
            if len(c_prime) < len(c):
                c = list(c_prime)
            else:
                break

        return oriented_edges


class ForestAugmentedNaiveBayesSearch(StructureEstimator):

    def __init__(self, data, class_node, root_node=None, **kwargs):
        '''
        Search class for learning forest-augmented naive bayes (FAN) graph structure with a given set of variables.
        FAN is an extension of Naive Bayes classifer and allows a forest structure over the independent variables
        to account for interaction.
        See http://www.cs.unb.ca/~hzhang/publications/DASFAA05-final.pdf for reference.
        '''
        self.class_node = class_node
        self.root_node = root_node

        super().__init__(data, **kwargs)

    def estimate(self):
        '''
        Estimates the DAG structure that fits best to the given data set using the Chow-Liu algorithm.
        Only estimates network structure, no parametrization.
        '''
        if self.class_node not in self.data.columns:
            raise ValueError("Class node must exist in data")

        if self.root_node is not None and self.root_node not in self.data.columns:
            raise ValueError("Root node must exist in data")

        graph = nx.Graph()
        df_features = self.data.loc[:, self.data.columns != self.class_node]
        total_cols = len(df_features.columns)
        cmis = []
        for i in range(total_cols):
            from_node = df_features.columns[i]
            graph.add_node(from_node)
            for j in range(i + 1, total_cols):
                to_node = df_features.columns[j]
                graph.add_node(to_node)
                cmi = conditional_mutual_info_score(
                    df_features.iloc[:, i], df_features.iloc[:, j],
                    self.data.loc[:, self.class_node]
                )
                cmis.append(cmi)
                graph.add_edge(from_node, to_node, weight=cmi)
        cmi_avg = np.mean(cmis)
        tree = nx.maximum_spanning_tree(graph)

        if not self.root_node:
            root_node = df_features.columns[0]
            root_node_mi = mutual_info_score(
                df_features.iloc[:, 0], self.data.loc[:, self.class_node]
            )
            for i in range(1, total_cols):
                node = df_features.columns[i]
                mi = mutual_info_score(
                    df_features.iloc[:, i], self.data.loc[:, self.class_node]
                )
                if mi > root_node_mi:
                    root_node = node
                    root_node_mi = mi
            self.root_node = root_node

        digraph = nx.bfs_tree(tree, self.root_node)

        edges = list(digraph.edges)
        weights = nx.get_edge_attributes(tree, 'weight')
        for from_node, to_node in edges:
            key = (
                (from_node, to_node) if (from_node, to_node) in weights
                else (to_node, from_node)
            )
            if weights[key] < cmi_avg:
                digraph.remove_edge(from_node, to_node)

        for node in df_features.columns:
            digraph.add_edge(self.class_node, node)

        return DAG(digraph)
