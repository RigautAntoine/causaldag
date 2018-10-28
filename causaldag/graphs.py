import networkx as nx
from networkx.algorithms import simple_paths
from collections import defaultdict
from .utils import get_all_possible_sets

class Graph():
    """
    Represents a causal graph and is a wrapper around the networkx.Graph class
    """
    
    def __init__(self):
        # Initialize a simple networkx Graph (undirected)
        self.G = nx.Graph()
        
        # Initialize an edge orientation graph
        self._edge_orient_dict = defaultdict(lambda: defaultdict(str))
        
    def add_edge(self, v_a, v_b):
        """
        Add an (undirected) edge between `v_a` and `v_b`
        
        Params:
            v_a (str): node `a`
            v_b (str): node `b`
        """
        self.G.add_edge(v_a, v_b)
        self.set_edge_orientation(v_a, v_b, set_null=True)
    
    def edges(self):
        """
        Returns:
            list of tuples
        """
        return self.G.edges()
    
    def nodes(self):
        """
        Returns:
            list of str
        """
        return self.G.nodes()
    
    def get_neighbors(self, v):
        """
        Get adjacent nodes
        """
        return self.G.neighbors(v)
    
    def get_successors(self, v):
        """
        Get successors - adjacent nodes in that are caused by `v`
        """
        neighbors = self.get_neighbors(v)
        return [n for n in neighbors if self.get_edge_orientation(v, n) == n]
    
    def get_descendants(self, v):
        """
        Get all descendants of `v`
        """
        descendants = set()
        visited = set()
        to_visit = [v]
        
        while len(to_visit) > 0:
            current_node = to_visit.pop()
            visited.add(current_node)
            for successor in self.get_successors(current_node):
                descendants.add(successor)
                if successor not in visited:
                    to_visit.append(successor)
                    
        return list(descendants)
    
    def remove_edge(self, v_a, v_b):
        """
        Remove edge from the graph
        """
        self.G.remove_edge(v_a, v_b)
        del self._edge_orient_dict[v_b][v_a]
        del self._edge_orient_dict[v_a][v_b]
    
    def get_edge_orientation(self, v_a, v_b):
        return self._edge_orient_dict[v_a][v_b]
    
    def set_edge_orientation(self, v_a, v_b, set_null=False):
        """
        Set the orientation for edge `v_a` and `v_b`
        
        Params:
            v_a (str): node `a`
            v_b (str): node `b`
            set_null (bool): if true, then the edge is considered undirected. 
                If false (default), then the edge is considered directed from `a` to `b`
        """
        if set_null is False:
            self._edge_orient_dict[v_a][v_b] = v_b
            self._edge_orient_dict[v_b][v_a] = v_b
        elif set_null is True:
            self._edge_orient_dict[v_a][v_b] = None
            self._edge_orient_dict[v_b][v_a] = None
        else:
            raise ValueError('Wrong input: `set_null` expects bool')
            
    def get_all_paths(self, v_a, v_b):
        """
        Return all the paths (causal and non-causal alike) between `v_a` `and v_b`
        """
        return list(simple_paths.all_simple_paths(self.G, v_a, v_b))
    
    def __str__(self):
        s = ''
        for (v_a, v_b) in self.edges():
            s += '{} to {}, directed at {}\n'.format(v_a, v_b, self.get_edge_orientation(v_a, v_b))
        return s
            

        
class CausalDAG(Graph):
    """
    Class for Directed Acyclic Graph. Same as Graph above, except that all edges must be directed.
    """
    
    def __init__(self):
        super().__init__()
        
    def add_edge(self, v_a, v_b):
        self.G.add_edge(v_a, v_b)
        self.set_edge_orientation(v_a, v_b)
        
    def backdoor_criterion(self, v_a, v_b):
        """Find the sets Z that satisfy the backdoor criterion between X and Y
        
        Such set:
            Has no node which is a descendant of treament X
            and the set of nodes blocks all of the backdoor paths between X and Y
        """
        # Step 1: Find all nodes that are not descendants of treatment X
        # Step 1a: Find those nodes which are descendants of treatment  
        descendants = self.get_descendants(v_a)        
        
        # Step 1b: Build the list of nodes that are not descendants of treatment
        nondescendants = [v for v in self.nodes() if v not in list(descendants) and v != v_a]
        
        if len(nondescendants) == 0:
            raise ValueError('All nodes are descendants of treatment')
        
        # Step 1c: Get all possible conditioning sets from the list of non-descendants
        candidate_sets = get_all_possible_sets(nondescendants)
        
        # Step 2: Find all backdoor paths from treatment to outcome
        # In other words, paths which are non-causal and have an arrow pointing to treatment
        
        # Step 2a: Get all paths
        paths = self.get_all_paths(v_a, v_b)
        
        # Step 2b: Filter down to those with an arrow to treatment 
        backdoor_paths = []
        
        for p in paths:
            if self.get_edge_orientation(v_a, p[1]) == v_a:
                backdoor_paths.append(Path(nodes=p, graph=self))
        
        if len(backdoor_paths) == 0:
            return ()
        
        # Step 3: Identify candidate sets that dseparates all backdoor path
        z_sets = []
        for z in candidate_sets:
            if all([p.is_dseparated(z=z) for p in backdoor_paths]):
                z_sets.append(z)

        return z_sets


class Node():
    """
    Class representing a node
    
    Attributes:
        label (str): name of the node
        is_collider (bool): flag the node as a collider in a path or not
        is_conditioned (bool): flag the node as conditioned/blocked or not
    """
    def __init__(self, label, is_collider):
        self.label = label
        self.is_collider = is_collider
        self.is_conditioned = False
           
    def condition(self):
        self.is_conditioned = True
        
    def uncondition(self):
        self.is_conditioned = False
            
class Path():
    """
    Class represents a sequence-path of nodes
    
    Information
    List of nodes
    """
    def __init__(self, nodes, graph):
        """
        Params:
            nodes (list of str): the list of nodes in path
            graph (Graph or CausalDAG object): the graph object which the path is from
        """
        self.graph = graph
        self.raw_nodes = nodes
        self._nodes = []
        self._initialize_nodes()
        
    def _initialize_nodes(self):
        for i, node in enumerate(self.raw_nodes[1:-1]):
            previous_node = self.raw_nodes[i]
            next_node = self.raw_nodes[i+2]
         
            if (self.graph.get_edge_orientation(node, previous_node) == node) and (self.graph.get_edge_orientation(node, next_node) == node):
                is_collider = True
            else:
                is_collider = False 
            
            self._nodes.append(Node(label = node, is_collider = is_collider))
    
    def nodes(self):
        return self._nodes
    
    def is_dseparated(self, z=None):
        """
        Find if the path between x and y is d-separated given z
        """
        
        # Initialize all nodes as unconditioned
        for node in self.nodes():
            node.uncondition()
        
        # Condition on the basis of z
        if z is not None:
            for node in self.nodes():
                if node.label in z:
                    node.condition()
                    
        # If at least one noncollider is conditioned, then d-separated
        for node in self.nodes():
            if node.is_collider is False and node.is_conditioned is True:
                return True
            
        # Else if there is at least one collider than has not be conditioned, then d-separated
        for node in self.nodes():
            if node.is_collider is True and node.is_conditioned is False:
                return True
            
        # Else: return False.
        return False