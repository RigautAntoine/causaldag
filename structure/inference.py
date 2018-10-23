from .graphs import Graph
from .utils import get_all_possible_sets

class IC_star():
    
    def __init__(self, data, independence_test, vartypes=None):
        """
        Parameters:
        
        data (pandas.DataFrame): data to infer a causal diagram from
        """
        self.data = data
        self.independence_test = independence_test
        self.nodes = list(data.columns)
        self.vartypes = vartypes
        self.graph = None
        self.conditioning_sets = {}
        self.infer()
        
    def infer(self):
        
        # Step 1: Initialize a fully-connected undirect graph
        self._initialize_graph()
        # Step 2: Find all conditioning sets
        self._find_conditioning_sets()
        # Step 3: Among non-adjacent sets, find if a given common neighbor has 
        self._find_colliders()
        # Step 4: Apply recursion rules
        recursion_ongoing = True
        while recursion_ongoing is True:
            recursion_rule_1_ongoing = self._recursion_rule_1()
            recursion_rule_2_ongoing = self._recursion_rule_2()
            recursion_ongoing = recursion_rule_1_ongoing or recursion_rule_2_ongoing
            
        
    def _initialize_graph(self):
        """
        Creates self.graph - a list of edges represented a fully-connected undirected graphs
        """
        self.graph = Graph()
        
        for (v_a, v_b) in get_all_possible_sets(self.nodes, 2):
            self.graph.add_edge(v_a, v_b)
        
        
    def _find_conditioning_sets(self):
        """
        For each edge in the initialized self.graph, find progressively larger conditioning sets
        until we find a set that d-separates the edge. If we find such a set, remove the edge.
        Else continue. Store findings in self.conditioning_sets
        """
        edges = list(self.graph.edges())
        for (v_a, v_b) in edges:
            
            # Nodes that can be conditioned on
            conditioning_nodes = [n for n in self.graph.nodes() if n not in [v_a, v_b]]
            cn_max = len(conditioning_nodes)
            
            for v in conditioning_nodes: # Check Can clear
                assert v not in [v_a, v_b]
            
            # Going from 1-node to set to larger sets
            found = False
            for q in range(1, cn_max+1):
                
                # Iterate for each set:
                
                for z_set in get_all_possible_sets(conditioning_nodes, q):
                    
                    self.independence_test.fit(x = [v_a], y = v_b, z = list(z_set), data = self.data)
                    
                    if self.independence_test.is_independent():
                        # Update self.conditioning_sets
                        self.conditioning_sets[v_a, v_b] = z_set
                        # Remove the edge
                        self.graph.remove_edge(v_a, v_b)
                        # Break through the whole loop
                        found = True
                        break
                
                if found is True:
                    break
            
            
            
        
    def _find_colliders(self):
        """
        Find neighbors between two adjacents nodes that are not part of the conditioning set
        for these two nodes. Those are colliders between the two nodes.
        """
        
        # For each node
        for node in self.graph.nodes():
            
            # Find the neighbors
            neighbors = self.graph.get_neighbors(node)
            
            # For each of these neighbor:
            for neighbor in neighbors:
                # Get its own neighbors
                neighbor_neighbors = [v for v in self.graph.get_neighbors(neighbor) if v != node and v not in neighbors]
                if len(neighbor_neighbors) == 0:
                    # If no nonadjacent then skip
                    continue
                
                for nonadjacent in neighbor_neighbors:
                    # Get the conditioning set 
                    z_set = self.conditioning_sets.get((node, nonadjacent), None)
                    if z_set is None:
                        z_set = self.conditioning_sets[nonadjacent, node]
                        
                    if neighbor not in z_set:
                        self.graph.set_edge_orientation(node, neighbor)
                        self.graph.set_edge_orientation(nonadjacent, neighbor)
                        #print('Found collider: {} -> {} <- {}'.format(node, neighbor, nonadjacent))
    
    def _recursion_rule_1(self):
        """
        Apply recursion rule 1.
        Find set of nonadjacent nodes (a, b) for which the neighbor c has an arrow from a into c 
        and no arrow from b into c. Mark that edge from b to c as directed towards b.
        """
        arrow_added = False
        # For each node
        for node in self.graph.nodes():
            
            # Find the neighbors
            neighbors = self.graph.get_neighbors(node)
            
            # For each of these neighbor:
            for neighbor in neighbors:
                
                if self.graph.get_edge_orientation(node,neighbor) != neighbor:
                    # If the edge from a to c is not pointing to c, skip
                    continue
                
                # Get its own neighbors
                neighbor_neighbors = [v for v in self.graph.get_neighbors(neighbor) 
                                      if v != node and v not in neighbors]
                
                if len(neighbor_neighbors) == 0:
                    # If no nonadjacent then skip
                    continue
                
                for nonadjacent in neighbor_neighbors:
                    if self.graph.get_edge_orientation(nonadjacent,neighbor) is None:
                        self.graph.set_edge_orientation(neighbor, nonadjacent)
                        arrow_added = True
                        
        return arrow_added
    
    def _recursion_rule_2(self):
        """
        For every pair of adjacent node (a) and (b), if there exist a completely directed path from
        (a) to (b) then a -> b. 
        """
        arrow_added = False
        
        for v_a in self.nodes:
            # Find the adjacent neighbors to v_a
            neighbors = self.graph.get_neighbors(v_a)
            
            # If no neighbor, skip to the next node
            if len(neighbors) == 0:
                continue
                
            for v_b in neighbors:
                
                # If we already have a direction for that edge, skip on to the next adjacent neighbor
                if self.graph.get_edge_orientation(v_a,v_b) is not None:
                    continue
                
                # Find all the paths between v_a and v_b
                paths = self.graph.get_all_paths(v_a, v_b)
                
                # Find if at least one of these paths is composed only of arrows leading from v_a to v_b
                for path in paths:
                    
                    directed = True
                    
                    # Iterate through the pair of (x, y) neighboring nodes in the path
                    for (x, y) in zip(path[:-1], path[1:]):
                        # If you find that on the path, one of the links is wrong-headed, directed=False
                        if self.graph.get_edge_orientation(x,y) != y:
                            directed = False
                            break
                            
                    if directed is True:
                        self.graph.set_edge_orientation(v_a,v_b)
                        arrow_added = True
                        break
                        
        return arrow_added