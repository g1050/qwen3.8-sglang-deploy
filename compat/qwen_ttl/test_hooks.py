import unittest

from qwen_ttl import hooks


class Node:
    def __init__(self, name, tag):
        self.name, self._ttl = name, tag
        self.lru_prev, self.lru_next = [None], [None]


class LRU:
    """The pointer layout of the engine's UnifiedLRUList: head is MRU, eviction walks from tail."""
    _pt, mid = 0, None

    def __init__(self, nodes):
        self.head, self.tail = Node('head', None), Node('tail', None)
        self.head.lru_next[0], self.tail.lru_prev[0] = self.tail, self.head
        for node in nodes:                      # given in eviction order; first ends next to the tail
            self._add_node_after(self.head, node)

    def _add_node_after(self, prev, new):
        new.lru_prev[0], new.lru_next[0] = prev, prev.lru_next[0]
        prev.lru_next[0].lru_prev[0] = new
        prev.lru_next[0] = new

    def _remove_node(self, node):
        node.lru_prev[0].lru_next[0], node.lru_next[0].lru_prev[0] = node.lru_next[0], node.lru_prev[0]
        node.lru_prev[0] = node.lru_next[0] = None

    def eviction_order(self):
        out, node = [], self.tail.lru_prev[0]
        while node is not self.head:
            out.append(node.name)
            node = node.lru_prev[0]
        return out

    def forward_order(self):
        out, node = [], self.head.lru_next[0]
        while node is not self.tail:
            out.append(node.name)
            node = node.lru_next[0]
        return out


class ReorderTail(unittest.TestCase):
    def setUp(self):
        hooks.STATE.now = 1000.0
        hooks.STATE_ORDER = True
        self.valid, self.expired, self.beyond = (990, 1500, 3600), (600, 900, 300), (-7000, -6700, 300)

    def lru(self, spec):
        tags = {'v': self.valid, 'e': self.expired, 'b': self.beyond}
        return LRU([Node(name, tags[name[0]]) for name in spec])

    def test_rank_first_then_lru(self):
        lru = self.lru(['v1', 'e1', 'v2', 'b1', 'e2', 'v3'])
        order = [n.name for n in hooks.reorder_tail(lru)]
        self.assertEqual(order, ['b1', 'e1', 'e2', 'v1', 'v2', 'v3'])
        self.assertEqual(lru.eviction_order(), order)
        self.assertEqual(lru.forward_order(), order[::-1])

    def test_sorted_tail_is_untouched(self):
        lru = self.lru(['b1', 'e1', 'v1', 'v2'])
        before = lru.eviction_order()
        hooks.reorder_tail(lru)
        self.assertEqual(lru.eviction_order(), before)

    def test_only_the_window_moves(self):
        lru = self.lru(['v1', 'e1', 'v2', 'e2', 'v3'])
        hooks.reorder_tail(lru, window=3)
        self.assertEqual(lru.eviction_order(), ['e1', 'v1', 'v2', 'e2', 'v3'])

    def test_untagged_nodes_start_a_default_lease(self):
        lru = LRU([Node('n1', None), Node('e1', self.expired)])
        hooks.reorder_tail(lru)
        self.assertEqual(lru.eviction_order(), ['e1', 'n1'])

    def test_session_partitioned_list_keeps_native_order(self):
        lru = self.lru(['v1', 'e1'])
        lru.mid = object()
        self.assertEqual(hooks.reorder_tail(lru), [])
        self.assertEqual(lru.eviction_order(), ['v1', 'e1'])


class ReorderDefault(unittest.TestCase):
    def test_disabled_unless_requested(self):
        hooks.STATE_ORDER = False
        hooks.STATE.now = 1000.0
        lru = LRU([Node('v1', (990, 1500, 3600)), Node('e1', (600, 900, 300))])
        self.assertEqual(hooks.reorder_tail(lru), [])
        self.assertEqual(lru.eviction_order(), ['v1', 'e1'])


class Tree:
    def __init__(self, name, parent=None):
        self.name, self.parent, self.children = name, parent, {}
        if parent is not None:
            parent.children[name] = self


class EvictionChain(unittest.TestCase):
    def test_single_child_ancestors_follow_the_leaf(self):
        root = Tree('root')
        prompt = Tree('prompt', root)
        turn = Tree('turn', prompt)
        tail = Tree('tail', turn)
        self.assertEqual([n.name for n in hooks.eviction_chain(tail, root)], ['prompt', 'turn', 'tail'])

    def test_stops_below_a_branching_ancestor(self):
        root = Tree('root')
        shared = Tree('shared', root)
        a, _ = Tree('a', shared), Tree('b', shared)
        leaf = Tree('leaf', a)
        self.assertEqual([n.name for n in hooks.eviction_chain(leaf, root)], ['a', 'leaf'])


class Data:
    def __init__(self, host_value):
        self.host_value = host_value


class Restorable(unittest.TestCase):
    def chain(self, states):
        nodes = []
        for has_state in states:
            node = Tree(str(len(nodes)))
            node.key, node.component_data = [0] * 64, {7: Data(object() if has_state else None)}
            nodes.append(node)
        return nodes

    def tearDown(self):
        hooks.STATE.state_component = None

    def test_chain_must_end_on_a_node_with_state(self):
        hooks.STATE.state_component = 7
        self.assertEqual(hooks.restorable(self.chain([False, True, False])), 2)
        self.assertEqual(hooks.restorable(self.chain([True, True])), 2)
        self.assertEqual(hooks.restorable(self.chain([False, False])), 0)

    def test_stateless_models_write_everything(self):
        self.assertEqual(hooks.restorable(self.chain([False, False])), 2)


if __name__ == '__main__':
    unittest.main()
