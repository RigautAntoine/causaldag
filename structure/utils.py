import itertools

def get_all_possible_sets(nodes, k=None, include_empty=False):
    """
    Get all possible sets from a given list of nodes
    """
    if k is None:
        k = len(nodes)+1
        sets = [z for i in range(1, k) for z in itertools.combinations(nodes, i)]
    else:
        sets = [z for z in itertools.combinations(nodes, k)]
    
    if include_empty:
        sets.append(()) 
    
    return sets