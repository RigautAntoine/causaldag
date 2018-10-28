from .utils import get_all_possible_sets
from .graphs import Path

class Implications():
    """
    Class that validates a user-provided Causal DAG against the data provided
    
    Attributes:
        graph (CausalDAG objet)
        implications: all testable implications of the provided graph
        weak_contradictions: the graph implied dependence but no dependence was detected by the test
        strong_contradictions: the graph implied no dependence between two variables 
            but the test found statistically-significant dependance
    """
    def __init__(self, graph, data, independence_test, categorical_vars=None):
        """
        Parameters:
            graph (CausalDAG object): user-provided DAG
            data (pandas.DataFrame): data to infer a causal diagram from
            independence_test (test object): method for testing independence between variables
            categorical_vars (list of str): list of categorical variables
        """
        
        self.graph = graph
        self.data = data
        if categorical_vars is None:
            self.categorical_vars = []
        else:
            self.categorical_vars = categorical_vars
        self.independence_test = independence_test
        
        # Step 1: Generate all testable implications
        self.implications = self._generate_testable_implications()
        
        # Step 2: For each implication, test if True
        self._check_implications_against_data()
        
        
    def _generate_testable_implications(self):
        independence_implications = []
        
        for (v_a, v_b) in get_all_possible_sets(self.graph.nodes(), 2):
            
            # Get all paths between v_a and v_b
            paths = [Path(nodes=p, graph=self.graph) for p in self.graph.get_all_paths(v_a, v_b)]
            
            # Get all nodes that can be controlled
            control_nodes = [n for n in self.graph.nodes() if n not in [v_a, v_b]]
            
            # All possible combinations of nodes
            z_sets = get_all_possible_sets(control_nodes, include_empty=True)
            
            for z in z_sets:
                # if Z d-separates all the possible paths, then it d-separate v_a and v_b
                if all([p.is_dseparated(z) for p in paths]):
                    independence_implications.append((v_a, v_b, list(z), True))
                else:
                    independence_implications.append((v_a, v_b, list(z), False))
                    
        return independence_implications
    
    def _check_implications_against_data(self):
        self.agreements = []
        self.weak_contradictions = []
        self.strong_contradictions = []
        
        for (x, y, z, independence_flag) in self.implications:
            
            if y in self.categorical_vars:
                categorical_outcome=True
            else:
                categorical_outcome=False
            
            self.independence_test.fit([x], y, z, categorical_outcome=categorical_outcome, data=self.data)
            test_result = self.independence_test.is_independent()
            
            if test_result == independence_flag:
                self.agreements.append((x, y, z, independence_flag))
            elif test_result is True:
                self.weak_contradictions.append((x, y, z, independence_flag))
            elif test_result is False:
                self.strong_contradictions.append((x, y, z, independence_flag))