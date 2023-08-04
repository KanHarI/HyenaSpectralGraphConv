from spectral_graph_conv.dataset.toy_undirected_tree import ToyUndirectedTree

NUM_TESTS = 10
VOCAB_SIZE = 10
N_NODES = 100


def test_parents_random_tree() -> None:
    for i in range(NUM_TESTS):
        tree = ToyUndirectedTree.create_random_tree(VOCAB_SIZE, N_NODES)
        parent_indices = tree.get_parent_node_indices()
        parent_labels = tree.get_parent_node_labels_tensor(VOCAB_SIZE)
        assert parent_indices[0] == -1
        assert parent_labels[0] == VOCAB_SIZE
        for i in range(1, N_NODES):
            assert parent_indices[i] < i
            assert parent_labels[i] == tree.nodes[parent_indices[i]]
            assert tree.edges[parent_indices[i], i]
