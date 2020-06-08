import networkx as nx
import numpy as np
from pgmpy.estimators import StructureEstimator
from sklearn.metrics import mutual_info_score
from pgmpy.base import DAG


class TreeAugmentedNaiveBayesSearch(StructureEstimator):
    def __init__(self, data, class_node, root_node=None, **kwargs):
        """
        Search class for learning tree-augmented naive bayes (TAN) graph structure with a given set of variables.
        TAN is an extension of Naive Bayes classifer and allows a tree structure over the independent variables
        to account for interaction.
        Parameters
        ----------
        data: pandas DataFrame object
            dataframe object where each column represents one variable.
        class_node: node
            Dependent variable of the model (i.e. the class label to predict)
        root_node: node (optional)
            The root node of the tree structure over the independent variables.  If not specified, then
            an arbitrary independent variable is selected as the root.
        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.
        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        References
        ----------
        Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network classifiers. Machine Learning 29: 131–163
        """
        self.class_node = class_node
        self.root_node = root_node

        super(TreeAugmentedNaiveBayesSearch, self).__init__(data, **kwargs)

    def estimate(self):
        """
        Estimates the `DAG` structure that fits best to the given data set using the chow-liu algorithm.
        Only estimates network structure, no parametrization.
        Returns
        -------
        model: `DAG` instance
            A tree augmented naive bayes `DAG`.
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from pgmpy.estimators import ExhaustiveSearch
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'E'])
        >>> class_node = 'A'
        >>> est = TreeAugmentedNaiveBayesSearch()
        >>> model = est.estimate()
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)
        >>> plt.show()
        """

        if self.class_node not in self.data.columns:
            raise ValueError("class node must exist in data")

        if self.root_node is not None and self.root_node not in self.data.columns:
            raise ValueError("root node must exist in data")

        # construct maximum spanning tree
        G = nx.Graph()
        df_features = self.data.loc[:, self.data.columns != self.class_node]
        total_cols = len(df_features.columns)
        for i in range(total_cols):
            from_node = df_features.columns[i]
            G.add_node(from_node)
            for j in range(i+1, total_cols):
                to_node = df_features.columns[j]
                G.add_node(to_node)
                # edge weight is the MI between a pair of independent variables
                mi = mutual_info_score(
                    df_features.iloc[:, i], df_features.iloc[:, j])
                G.add_edge(from_node, to_node, weight=mi)
        T = nx.maximum_spanning_tree(G)

        # create DAG by directing edges away from root node
        if self.root_node:
            D = nx.bfs_tree(T, self.root_node)
        else:
            D = nx.bfs_tree(T, df_features.columns[0])

        # add edge from class node to other nodes
        for node in df_features.columns:
            D.add_edge(self.class_node, node)

        return DAG(D)


class BNAugmentedNaiveBayesSearch(StructureEstimator):
    def __init__(self, data, class_node, root_node=None, **kwargs):
        """
        Search class for learning tree-augmented naive bayes (TAN) graph structure with a given set of variables.
        TAN is an extension of Naive Bayes classifer and allows a tree structure over the independent variables
        to account for interaction.
        Parameters
        ----------
        data: pandas DataFrame object
            dataframe object where each column represents one variable.
        class_node: node
            Dependent variable of the model (i.e. the class label to predict)
        root_node: node (optional)
            The root node of the tree structure over the independent variables.  If not specified, then
            an arbitrary independent variable is selected as the root.
        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.
        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        References
        ----------
        Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network classifiers. Machine Learning 29: 131–163
        """
        self.class_node = class_node
        self.root_node = root_node

        super(BNAugmentedNaiveBayesSearch, self).__init__(data, **kwargs)

    def estimate(self, epsilon=0.1):
        """
        Estimates the `DAG` structure that fits best to the given data set using the chow-liu algorithm.
        Only estimates network structure, no parametrization.
        Returns
        -------
        model: `DAG` instance
            A tree augmented naive bayes `DAG`.
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from pgmpy.estimators import ExhaustiveSearch
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'E'])
        >>> class_node = 'A'
        >>> est = TreeAugmentedNaiveBayesSearch()
        >>> model = est.estimate()
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)
        >>> plt.show()
        """

        if self.class_node not in self.data.columns:
            raise ValueError("class node must exist in data")

        if self.root_node is not None and self.root_node not in self.data.columns:
            raise ValueError("root node must exist in data")

        ################## Drafting #####################

        L = []
        graph = nx.Graph()
        df_features = self.data.loc[:, self.data.columns != self.class_node]
        total_cols = len(df_features.columns)
        graph.add_nodes_from(df_features.columns)
        for i in range(total_cols):
            from_node = df_features.columns[i]
            for j in range(i + 1, total_cols):
                to_node = df_features.columns[j]
                mi = mutual_info_score(
                    df_features.iloc[:, i], df_features.iloc[:, j]
                )
                if mi < epsilon:
                    L.append((from_node, to_node, mi))

        # Sort pairs of nodes by decreasing mutual information
        L.sort(key=lambda tup: tup[2], reverse=True)

        # Get the first two pairs of nodes and add corresponding edges
        from_node, to_node, mi = L.pop(0)
        graph.add_edge(from_node, to_node, weight=mi)
        from_node, to_node, mi = L.pop(0)
        graph.add_edge(from_node, to_node, weight=mi)

        # If there is no adjacency path between a pair of nodes, add the corresponding edge
        for i in range(len(L)):
            from_node, to_node, mi = L[i]
            if len(graph.edges) - 1 == len(graph.nodes):
                break
            if not from_node in graph.neighbors(to_node):
                graph.add_edge(from_node, to_node, weight=mi)
                L.pop(i)

        ################## Thickening ###################

        for i in range(len(L)):
            from_node, to_node, mi = L[i]
            if not self.try_to_separate_a(graph, from_node, to_node, epsilon):
                graph.add_edge(from_node, to_node, weight=mi)

        ################## Thinning #####################

        for edge in graph.edges.data():
            from_node, to_node, _ = edge
            graph.remove_edge(edge)
            if nx.has_path(graph, from_node, to_node):
                if not self.try_to_separate_a(graph, from_node, to_node, epsilon):
                    graph.add_edge(edge)

        for edge in graph.edges.data():
            from_node, to_node, _ = edge
            graph.remove_edge(edge)
            if nx.has_path(graph, from_node, to_node):
                if not self.try_to_separate_b(graph, from_node, to_node):
                    graph.add_edge(edge)

        self.orient_edges(graph)

    def try_to_separate_a(self, graph, from_node, to_node, epsilon=0.1):
        from_node_neighbors = nx.neighbors(graph, from_node)
        to_node_neighbors = nx.neighbors(graph, to_node)
        n1 = []
        n2 = []
        for path in nx.all_simple_paths(graph, source=from_node, target=to_node):
            for node in path:
                if node in from_node_neighbors:
                    n1.append(node)
                if node in to_node_neighbors:
                    n2.append(node)

        # Remove the currently known child-nodes of node1 from N1
        # and child-nodes of node2 from N2 (?)

        if len(n1) > len(n2):
            n1, n2 = n2, n1

        skip_step = False
        n2_used = False
        c = list(n1)
        while True:
            if not skip_step:
                v = self.conditional_mutual_info_score(from_node, to_node, c)
                if v < epsilon:
                    return True

            if len(c) == 1:
                if not n2_used:
                    c = list(n2)
                    n2_used = True
                else:
                    return False
            else:
                values = []
                conditions = []
                for node in graph.nodes:
                    ci = list(c) - [node]
                    vi = self.conditional_mutual_info_score(
                        from_node, to_node, ci
                    )
                min_index = np.argmin(values)
                vm = values[min_index]
                cm = conditions[min_index]
                if vm < epsilon:
                    return True
                elif vm > v:
                    if not n2_used:
                        c = list(n2)
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

    def try_to_separate_b(self, graph, from_node, to_node, epsilon=0.1):
        from_node_neighbors = nx.neighbors(graph, from_node)
        to_node_neighbors = nx.neighbors(graph, to_node)
        n1 = []
        n2 = []
        paths = nx.all_simple_paths(graph, source=from_node, target=to_node)
        for path in paths:
            for node in path:
                if node in from_node_neighbors:
                    n1.append(node)
                if node in to_node_neighbors:
                    n2.append(node)

        n1_prime = []
        for node in n1:
            neighbors = nx.neighbors(graph, node)
            for neighbor in neighbor:
                if neighbor not in n1:
                    n1_prime.append(neighbor)

        n2_prime = []
        for node in n2:
            neighbors = nx.neighbors(graph, node)
            for neighbor in neighbor:
                if neighbor not in n2:
                    n2_prime.append(neighbor)

        if len(n1) + len(n1_prime) < len(n2) + len(n2_prime):
            c = list(n1)
            c.extend(list(n1_prime))
        else:
            c = list(n2)
            c.extend(list(n2_prime))

        while True:
            v = self.conditional_mutual_info_score(from_node, to_node, c)
            if v < epsilon:
                return True

            c_prime = list(c)
            values = []
            conditions = []
            for node in graph.nodes:
                ci = list(c) - [node]
                vi = self.conditional_mutual_info_score(
                    from_node, to_node, ci
                )
                if vi < epsilon:
                    return True
                elif vi <= v + epsilon:
                    c_prime.remove(node)

            if len(c_prime) < len(c):
                c = list(c_prime)
            else:
                return False

        return False

    def orient_edges(self, graph):
        pass

    def conditional_mutual_info_score(self, xi, xj, c):
        cond_mutual_info = 0
        for cond in c:
            unique_condition_values = np.unique(cond)
            for i in unique_condition_values:
                condition_proba = np.sum(cond == i) / len(cond)
                cond_mutual_info += mutual_info_score(
                    xi[cond == i], xj[cond == i],
                ) * condition_proba
        return np.sum(cond_mutual_info)
