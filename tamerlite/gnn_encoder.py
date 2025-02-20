from fractions import Fraction
from typing import List
import networkx as nx
import unified_planning as up
import matplotlib.pyplot as plt

from unified_planning.model.walkers import Dnf
from tamerlite.core import get_fluent_value


class BasicEmbedding:
    def __init__(self, start=0) -> None:
        self._map = {}
        self._start = start
        self._debug = {}

    def __call__(self, x):
        res = self._map.setdefault(x, len(self._map) + self._start)
        self._debug[res] = x
        return res

class PTGraph:
    NODE_ATTRIBUTES = ['kind', 'type', 'fluent', 'value', 'goal_value', 'step', 'action']
    EDGE_ATTRIBUTES = ['ith', 'bound', 'pos']

    def __init__(self):
        self.nodes = []
        self.edges = [], []
        self.edge_features = []
        self.node_names = {}

    def add_node(self, name, **attributes):
        next_id = len(self.nodes)
        id = self.node_names.setdefault(name, next_id)
        if id == next_id:
            r = [attributes.get(k, 0) for k in self.NODE_ATTRIBUTES]
            self.nodes.append(r)
            return next_id
        else:
            for i, k in enumerate(self.NODE_ATTRIBUTES):
                v = attributes.get(k, None)
                if v is not None:
                    self.nodes[id][i] = v
            return id

    def add_edge(self, u, v, **attributes):
        self.edges[0].append(self.node_names[u])
        self.edges[1].append(self.node_names[v])
        r = [attributes.get(k, 0) for k in self.EDGE_ATTRIBUTES]
        self.edge_features.append(r)
        return len(self.edges[0])

    def clone(self):
        res = PTGraph()
        res.nodes = list(self.nodes)
        res.edges = list(self.edges[0]), list(self.edges[1])
        res.edge_features = list(self.edge_features)
        res.node_names = dict(self.node_names)
        return res

    def has_node(self, name):
        return name in self.node_names

    def to_networkx(self):
        invert = {v: k for k, v in self.node_names.items()}
        print(invert)
        res = nx.DiGraph()
        for i, feats in enumerate(self.nodes):
            attrs = {k: v for k, v in zip(self.NODE_ATTRIBUTES, feats)}
            res.add_node(invert[i], **attrs)
        for i, (u, v) in enumerate(zip(*self.edges)):
            attrs = {k: v for k, v in zip(self.EDGE_ATTRIBUTES, self.edge_features[i])}
            res.add_edge(invert[u], invert[v], **attrs)
        return res

    def plot(self, node_mapping):
        nxG = self.to_networkx()
        kind2color = {1 : 'red', 2 : 'blue', 3 : 'green', 4 : 'yellow'}
        node_colors = [kind2color[nxG.nodes[n]['kind']] for n in nxG.nodes]

        labels = {}
        for n in nxG.nodes:
            try:
                labels[n] = str(node_mapping._debug[n])
                #print(f" {n} -> {nxG.nodes[n]['kind']} -> {labels[n]} -> {node_colors[n]}")
            except:
                labels[n] = str(n)

        nx.draw_spring(nxG, labels=labels, with_labels=True, node_color=node_colors, node_size=10000)
        # # print(self.G.nodes(data=True))
        plt.show()
        #exit(0)


class GNNStateEncoder:

    OBJECT_KIND = 1
    FLUENT_KIND = 2
    ACTION_KIND = 3
    TN_EVENT_KIND = 4

    def __init__(self, search_space, grounding_result, initial_values, goals):
        self._grounding_result = grounding_result
        self._initial_values = initial_values

        problem = grounding_result.problem
        self._problem = problem

        self.search_space = search_space
        environment = problem.environment
        self._environment = environment

        self.G = PTGraph()

        self.node_mapping = BasicEmbedding(start=0)
        self.types_embedding = BasicEmbedding(start=1)
        self.fluent_embedding = BasicEmbedding(start=1)
        self.action_embedding = BasicEmbedding(start=1)

        self._lifted_action_cache = {}

        # # First, we create a node for each object, the attributes encodes the type
        # for o in problem.all_objects:
        #     self._add_object_node(graph=self.G, object=o)

        # Then each static fluent is liked to the objects it uses as params
        for fe in problem.get_static_fluents():
            v = initial_values[fe]
            self._add_fluent_node(graph=self.G, fluent_expr=fe, value=v)

        # encoding of the goal
        w = Dnf(environment)
        goals = w.walk(environment.expression_manager.And(goals))[0]
        for g in goals:
            if g.is_fluent_exp():
                self._add_fluent_node(graph=self.G, fluent_expr=g, goal_value=environment.expression_manager.TRUE())
            elif g.is_not():
                self._add_fluent_node(graph=self.G, fluent_expr=g.arg(0), goal_value=environment.expression_manager.FALSE())
            elif g.is_equals() and g.arg(0).is_fluent_exp():
                self._add_fluent_node(graph=self.G, fluent_expr=g.arg(0), goal_value=g.arg(1))
            elif g.is_equals() and g.arg(1).is_fluent_exp():
                self._add_fluent_node(graph=self.G, fluent_expr=g.arg(1), goal_value=g.arg(0))
            else:
                raise NotImplementedError

    def _to_lifted(self, action_name):
        action, params = self._lifted_action_cache.get(action_name, (None, None))
        if action is None:
            lifted_action_instance = self._grounding_result.map_back_action_instance(self._grounding_result.problem.action(action_name)())
            action = lifted_action_instance.action
            params = lifted_action_instance.actual_parameters
            self._lifted_action_cache[action_name] = (action, params)
        return (action, params)


    def _add_object_node(self, graph, object):
        no = self.node_mapping(object)
        if not graph.has_node(no):
            ty = self.types_embedding(object.type)
            graph.add_node(no, kind=self.OBJECT_KIND, type=ty)
        return no


    def _add_fluent_node(self, graph, fluent_expr, value=None, goal_value=None):
        nm = self.node_mapping
        nfe = nm(fluent_expr)
        is_new_node = not graph.has_node(nfe)

        if is_new_node:
            ty = self.types_embedding(fluent_expr.fluent().type)
            fl = self.fluent_embedding(fluent_expr.fluent())
            graph.add_node(nfe, kind=self.FLUENT_KIND, type=ty, fluent=fl)

        attributes = {}
        if fluent_expr.fluent().type.is_user_type():
            if value is not None:
                attributes["value"] = float(True)
            if goal_value is not None:
                attributes["goal_value"] = float(True)
        else:
            if value is not None:
                attributes["value"] = float(value.constant_value())
            if goal_value is not None:
                attributes["goal_value"] = float(goal_value.constant_value())

        graph.add_node(nfe, **attributes)
        if is_new_node:
            for i, a in enumerate(fluent_expr.args):
                graph.add_edge(nfe, self._add_object_node(graph, a.object()), ith=i)
            if fluent_expr.fluent().type.is_user_type():
                if value is not None:
                    graph.add_edge(nfe, self._add_object_node(graph, value.object()), ith=-1)
                if goal_value is not None:
                    graph.add_edge(nfe, self._add_object_node(graph, goal_value.object()), ith=-2)
        return nfe


    def _add_action_node(self, graph, action, params, step=0):
        nid = self.node_mapping((action.name, params))
        if not graph.has_node(nid):
            act = self.action_embedding(action.name)
            graph.add_node(nid, kind=self.ACTION_KIND, step=step, action=act)
            for i, p in enumerate(params):
                graph.add_edge(nid, self._add_object_node(graph, p.object()), ith=i)
        return nid

    def _add_tn_node(self, graph, v):
        nid = self.node_mapping(v)
        if not graph.has_node(nid):
            if len(v) == 3:
                action_name, is_start, _ = v
                pos = -1 if is_start else -2
            else:
                assert len(v) == 2
                event = v[0]
                action_name = event.action
                pos = event.pos

            action, params = self._to_lifted(action_name)
            act = self.action_embedding(action.name)
            graph.add_node(nid, kind=self.TN_EVENT_KIND, step=pos, action=act)
            aid = self._add_action_node(graph=graph, action=action, params=params)
            graph.add_edge(nid, aid, pos=pos)
        return nid

    def _add_tn_edge(self, graph, u, v, bound):
        iv = self._add_tn_node(graph, v)
        iu = self._add_tn_node(graph, u)
        graph.add_edge(iv, iu, bound=bound)


    def get_state_as_graph(self, state):
        G = self.G.clone()

        # Encode fluents (mu)
        for fe in self._initial_values.keys():
            py_v = get_fluent_value(str(fe), state)
            v = None
            if type(py_v) == bool:
                v = self._environment.expression_manager.Bool(py_v)
            elif type(py_v) == int:
                v = self._environment.expression_manager.Int(py_v)
            elif type(py_v) == Fraction:
                v = self._environment.expression_manager.Real(py_v)
            elif type(py_v) == str:
                o = self._problem.object(py_v)
                v = self._environment.expression_manager.ObjectExp(o)
            self._add_fluent_node(graph=G, fluent_expr=fe, value=v)

        # Encode actions (lambda)
        for gaction_name, (x, _) in state.todo.items():
            action, params = self._to_lifted(gaction_name)
            self._add_action_node(graph=G, action=action, params=params, step=x)

        # Encode Path
        previous = None
        for (gaction_name, pos, event_id) in state.path:
            nid = self.node_mapping((gaction_name, pos, event_id))

            action, params = self._to_lifted(gaction_name)
            act = self.action_embedding(action.name)
            G.add_node(nid, kind=self.TN_EVENT_KIND, step=pos, action=act)
            aid = self._add_action_node(graph=G, action=action, params=params)
            G.add_edge(nid, aid, pos=pos)

            if previous is not None:
                G.add_edge(previous, nid)
            previous = nid

        #G.plot(self.node_mapping)

        return G

