import time
import numpy as np
import torch
from typing import List, Tuple

import torch.nn as nn

from parser import sql_parser
from common.workload import Node, NodeOps, WorkloadInfo
from common.base_config import BaseConfig

# Import from local package
from ..encoders import plan_graph_encoder


class Fringe:
    """Priority queue and bookkeeping for beam search, equivalent to the original's open/expanded + fringe logic."""

    def __init__(self):
        self.fringe = []  # list[(cost, state)]， nodes not explored yet
        self.states_open = {}  # hash -> cost, nodes not explored yet, for fast access
        self.states_expanded = {}  # hash -> cost, nodes that have been explored

    @staticmethod
    def _hash_state(state: List):
        return hash(frozenset([subplan.to_str(with_cost=False) for subplan in state]))

    def add(self, cost, state):
        """Add a new state if it's not already in open/expanded."""
        h = self._hash_state(state)
        if h in self.states_open or h in self.states_expanded:
            # already seen, skip
            return
        self.states_open[h] = cost
        self.fringe.append((cost, state))

    def pop_best(self):
        """Pop the state with the smallest cost from fringe, and move it from open to expanded."""
        self.fringe = sorted(self.fringe, key=lambda x: x[0])
        # move the best node from un-explored to explored
        cost, state = self.fringe.pop(0)
        h = self._hash_state(state)
        self.states_open.pop(h, None)
        self.states_expanded[h] = cost
        return cost, state

    def remove_from_open(self, cost, state):
        """For epsilon-greedy branch, discard other fringe elements and sync removal from open (equivalent to original)."""
        h = self._hash_state(state)
        self.states_open.pop(h, None)


class PlanEnumerator:

    @staticmethod
    def get_possible_plans(query_node: Node,
                           state,
                           join_graph,
                           planner_config=None,
                           avoid_eq_filters=False,
                           workload_info=None,
                           use_plan_restrictions=True):
        # if not bushy:
        #     if planner_config and planner_config.search_space == 'leftdeep':
        #         return PlanEnumerator._get_left_deep(
        #             query_node, state, join_graph,
        #             planner_config, avoid_eq_filters,
        #             workload_info, use_plan_restrictions
        #         )
        #     else:
        #         raise
        # else:
        return PlanEnumerator._get_bushy_join_plans(
            query_node, state, join_graph,
            planner_config, avoid_eq_filters,
            workload_info, use_plan_restrictions
        )

    @staticmethod
    def _get_bushy_join_plans(query_node, state: List[Node], join_graph,
                              planner_config=None, avoid_eq_filters=False,
                              workload_info=None, use_plan_restrictions=True) -> List[Tuple[Node, int, int]]:
        """
        enumerates all valid binary join extensions (covering join algorithms and scan variants) from the current state,
        producing the next-level candidate plans
        """
        possible_joins = []
        num_rels = len(state)
        for i in range(num_rels):
            for j in range(num_rels):
                if i == j:
                    continue
                l, r = state[i], state[j]
                if not NodeOps.exists_join_edge_in_graph(l, r, join_graph):
                    continue
                for plan in PlanEnumerator._enumerate_ops(
                        l, r, workload_info, use_plan_restrictions,
                        planner_config, avoid_eq_filters
                ):
                    possible_joins.append((plan, i, j))
        return possible_joins

    # @staticmethod
    # def _get_left_deep(query_node, state, join_graph,
    #                    planner_config=None, avoid_eq_filters=False,
    #                    workload_info=None, use_plan_restrictions=True):
    #     possible_joins = []
    #     num_rels = len(state)
    #     join_index = None
    #     for i, s in enumerate(state):
    #         if s.IsJoin():
    #             assert join_index is None, 'two joins found'
    #             join_index = i
    #
    #     if join_index is None:
    #         # Base state: equivalent to original, use dedup set to avoid duplicate (i,j), but allow both directions
    #         scored = set()
    #         for i in range(num_rels):
    #             for j in range(num_rels):
    #                 if i == j:
    #                     continue
    #                 if (i, j) in scored:
    #                     continue
    #                 scored.add((i, j))
    #                 l, r = state[i], state[j]
    #                 if not NodeOps.exists_join_edge_in_graph(l, r, join_graph):
    #                     continue
    #                 for plan in PlanEnumerator._enumerate_ops(
    #                         l, r, workload_info, use_plan_restrictions,
    #                         planner_config, avoid_eq_filters
    #                 ):
    #                     possible_joins.append((plan, i, j))
    #     else:
    #         i, l = join_index, state[join_index]
    #         for j in range(num_rels):
    #             if j == i:
    #                 continue
    #             r = state[j]
    #             if not NodeOps.exists_join_edge_in_graph(l, r, join_graph):
    #                 continue
    #             for plan in PlanEnumerator._enumerate_ops(
    #                     l, r, workload_info, use_plan_restrictions,
    #                     planner_config, avoid_eq_filters
    #             ):
    #                 possible_joins.append((plan, i, j))
    #     return possible_joins

    @staticmethod
    def _enumerate_ops(left, right, workload_info: WorkloadInfo, use_plan_restrictions,
                       planner_config=None, avoid_eq_filters=False):
        # Equivalent to original: get join/scan types from BMOptimizer.workload_info, and pass use_plan_restrictions
        join_ops = workload_info.join_types
        scan_ops = workload_info.scan_types
        # if planner_config:
        #     join_ops = planner_config.KeepEnabledJoinOps(join_ops)
        return NodeOps.enumerate_join_with_ops(
            left, right,
            join_ops=join_ops, scan_ops=scan_ops,
            avoid_eq_filters=avoid_eq_filters,
            use_plan_restrictions=use_plan_restrictions
        )


class BMOptimizer:
    """Creates query execution plans using learned model. (Equivalent behavior to original)"""

    def __init__(self,
                 workload_info: WorkloadInfo,
                 plan_featurizer,
                 parent_pos_featurizer,
                 query_featurizer,
                 inverse_label_transform_fn,
                 model: nn.Module,
                 plan_physical=False,
                 use_plan_restrictions=True):

        # tree_conv = True
        # Same defaults for beam and termination as original
        beam_size = 20
        search_until_n_complete_plans = 10
        self.minimal_predicated_cost = -1e30
        self.workload_info = workload_info
        self.plan_featurizer = plan_featurizer
        self.parent_pos_featurizer = parent_pos_featurizer
        self.query_featurizer = query_featurizer
        self.inverse_label_transform_fn = inverse_label_transform_fn
        self.use_label_cache = True
        self.use_plan_restrictions = use_plan_restrictions

        # Equivalent defensive assertions as original (only 'Join' / 'Scan' in logical mode)
        if not plan_physical:
            jts = workload_info.join_types
            assert np.array_equal(jts, ['Join']), jts
            sts = workload_info.scan_types
            assert np.array_equal(sts, ['Scan']), sts

        self.plan_physical = plan_physical
        self.beam_size = beam_size
        self.search_until_n_complete_plans = search_until_n_complete_plans
        # self.tree_conv = tree_conv

        self.value_network = None
        self.label_cache = {}
        self.set_model_for_eval(model)

        # Debug stats
        self.total_joins = 0
        self.total_random_triggers = 0
        self.num_queries_with_random = 0

    def set_model_for_eval(self, model):
        self.value_network = model.to(BaseConfig.DEVICE)
        self.value_network.eval()
        self.label_cache = {}

    # @profile
    def infer(self, query_node: Node, sub_plan_nodes: List[Node], set_model_eval: bool = False):
        """
        Forward inference fully consistent with original:
        - Supports label cache
        - If tree_conv or plan_featurizer has pad attr -> use three-arg path
        - Else use Dataset two-arg path
        """
        labels = [None] * len(sub_plan_nodes)
        plans, idx, lookup_keys = [], [], []

        if self.use_label_cache:
            lookup_keys = [(query_node.info['query_name'], p.to_str(with_cost=False)) for p in sub_plan_nodes]
            for i, key in enumerate(lookup_keys):
                cached = self.label_cache.get(key)
                if cached is not None:
                    labels[i] = cached
                else:
                    plans.append(sub_plan_nodes[i])
                    idx.append(i)
            if len(plans) == 0:
                return labels
        else:
            plans = sub_plan_nodes

        if set_model_eval:
            self.value_network.eval()

        with torch.no_grad():
            query_enc = self.query_featurizer(query_node)
            all_query_vecs = [query_enc] * len(plans)

            all_plans, all_indexes = plan_graph_encoder.make_and_featurize_trees(
                node_featurizer=self.plan_featurizer, trees=plans)

            # all_plans = []
            # all_indexes = []
            # if self.tree_conv:
            #     all_plans, all_indexes = plan_graph_encoder.make_and_featurize_trees(
            #         plans, self.plan_featurizer)
            # else:
            #     for p in plans:
            #         all_plans.append(self.plan_featurizer(p))
            #     if self.parent_pos_featurizer is not None:
            #         for p in plans:
            #             all_indexes.append(self.parent_pos_featurizer(p))

            # Three-arg path (consistent with original, non_blocking=True)
            query_feat = torch.from_numpy(np.asarray(all_query_vecs)).to(BaseConfig.DEVICE, non_blocking=True)
            plan_feat = torch.from_numpy(np.asarray(all_plans)).to(BaseConfig.DEVICE, non_blocking=True)
            pos_feat = torch.from_numpy(np.asarray(all_indexes)).to(BaseConfig.DEVICE, non_blocking=True)
            cost = self.value_network(query_feat, plan_feat, pos_feat).cpu().numpy()

            # if self.tree_conv or hasattr(self.plan_featurizer, 'pad'):
            #     # Three-arg path (consistent with original, non_blocking=True)
            #     query_feat = torch.from_numpy(np.asarray(all_query_vecs)).to(BaseConfig.DEVICE, non_blocking=True)
            #     plan_feat = torch.from_numpy(np.asarray(all_plans)).to(BaseConfig.DEVICE, non_blocking=True)
            #     pos_feat = torch.from_numpy(np.asarray(all_indexes)).to(BaseConfig.DEVICE, non_blocking=True)
            #     cost = self.value_network(query_feat, plan_feat, pos_feat).cpu().numpy()
            # else:
            #     # Two-arg path (consistent with original)
            #     all_costs = [1] * len(all_plans)
            #     batch = ds.PlansDataset(
            #         all_query_vecs,
            #         all_plans,
            #         all_indexes,
            #         all_costs,
            #         transform_cost=False,
            #         return_indexes=False,
            #     )
            #     loader = torch.utils.data.DataLoader(batch, batch_size=len(all_plans), shuffle=False)
            #     processed_batch = list(loader)[0]
            #     query_feat, plan_feat = processed_batch[0].to(BaseConfig.DEVICE), processed_batch[1].to(BaseConfig.DEVICE)
            #     cost = self.value_network(query_feat, plan_feat).cpu().numpy()

            cost = self.inverse_label_transform_fn(cost)
            plan_labels = cost.reshape(-1, ).tolist()

        if self.use_label_cache:
            for i, label in enumerate(plan_labels):
                labels[idx[i]] = label
                self.label_cache[lookup_keys[idx[i]]] = label
            return labels
        else:
            return plan_labels

    def _make_new_states(self, state: List[Node], costs: List[float], possible_joins: List[Tuple[Node, int, int]]):
        """
        Consistent with original:
        - Apply (join,i,j) to state, merging two subtrees
        - New state cost = max join subtree cost in current state
        """
        valid_costs, valid_states = [], []
        for (join, i, j), predicated_cost in zip(possible_joins, costs):
            join.cost = predicated_cost
            new_state = state[:]  # Shallow copy
            new_state[i] = join
            del new_state[j]
            new_cost = self.minimal_predicated_cost
            for rel in new_state:
                if rel.IsJoin():
                    new_cost = max(new_cost, rel.cost)
            valid_costs.append(new_cost)
            valid_states.append(new_state)
        return valid_costs, valid_states

    def plan(self,
             query_node: Node,
             return_all_found=False,
             planner_config=None,
             verbose=False,
             avoid_eq_filters=False,
             epsilon_greedy=0) -> Tuple[float, Node, float, list]:
        """
        return_all_found： Return all complete plans, not just the cheapest one.
        verbose： Enable detailed debug logging of search and plans.
        planner_config； Restrict search space (e.g., only certain join ops, left-deep trees).
        epsilon_greedy： Exploration probability: occasionally keep a random plan instead of the best.
        avoid_eq_filters： Avoid generating certain equality-filter joins (like info_type) that PG rejects.
        """

        # if planner_config:
        #     if bushy:
        #         assert planner_config.search_space == 'bushy', planner_config
        #     else:
        #         assert planner_config.search_space != 'bushy', planner_config

        planning_start_t = time.time()

        # Build join graph
        join_graph, _ = query_node.get_parse_update_sql(sql_parse_func=sql_parser.simple_parse_sql)
        # Leaves as initial state
        query_leaves = query_node.GetLeaves()
        init_state = query_leaves

        # Equivalent to original: fringe starts with (0, init_state), marked in open
        fringe = Fringe()
        fringe.add(0, init_state)

        terminal_states = []
        is_eps_greedy_triggered = False

        while len(terminal_states) < self.search_until_n_complete_plans and fringe.fringe:
            state_cost, state = fringe.pop_best()

            if len(state) == 1:
                # Complete plan
                terminal_states.append((state_cost, state))
                continue

            possible_plans = PlanEnumerator.get_possible_plans(
                query_node=query_node,
                state=state,
                join_graph=join_graph,
                planner_config=planner_config,
                avoid_eq_filters=avoid_eq_filters,
                workload_info=self.workload_info,
                use_plan_restrictions=self.use_plan_restrictions
            )
            costs = self.infer(query_node, [join_node for join_node, _, _ in possible_plans])
            valid_costs, valid_states = self._make_new_states(state, costs, possible_plans)

            for c, new_state in zip(valid_costs, valid_states):
                fringe.add(c, new_state)

            # exploration or not
            r = np.random.rand()
            if r < epsilon_greedy:
                # Consistent with original: randomly keep one in fringe, remove others from open
                rand_idx = np.random.randint(len(fringe.fringe))
                keep_item = fringe.fringe[rand_idx]
                for i, item in enumerate(fringe.fringe):
                    if i == rand_idx:
                        continue
                    _c, _s = item
                    fringe.remove_from_open(_c, _s)
                fringe.fringe = [keep_item]

                self.total_random_triggers += 1
                is_eps_greedy_triggered = True

            # Beam truncation, only keep the top few of the BFS
            fringe.fringe = sorted(fringe.fringe, key=lambda x: x[0])[:self.beam_size]

        planning_time = (time.time() - planning_start_t) * 1e3
        print('Planning took {:.1f}ms'.format(planning_time))

        # Select optimal
        all_found = []
        min_cost = np.min([c for c, s in terminal_states])
        min_cost_idx = np.argmin([c for c, s in terminal_states])
        for (cost, state) in terminal_states:
            all_found.append((cost, state[0]))
            if verbose:
                tag = '  <-- cheapest' if cost == min_cost else ''
                print('  {:.1f} {}{}'.format(
                    cost, str([s.hint_str(self.plan_physical) for s in state]), tag))

        ret = [planning_time, terminal_states[min_cost_idx][1][0], terminal_states[min_cost_idx][0]]
        if return_all_found:
            ret.append(all_found)
        else:
            ret.append([])

        self.total_joins += len(query_leaves) - 1
        self.num_queries_with_random += int(is_eps_greedy_triggered)

        # in format of [planning time, best predicated plan, best plan's cost]
        return ret

    def SampleRandomPlan(self, query_node, bushy=True):
        """Equivalent random sampling to original."""
        planning_start_t = time.time()
        join_graph, _ = query_node.get_parse_update_sql(sql_parse_func=sql_parser.simple_parse_sql)
        query_leaves = query_node.CopyLeaves()
        num_rels = len(query_leaves)
        num_random_plans = 1000  # Same final setting as original

        def _SampleOne(state):
            while len(state) > 1:
                possible_plans = PlanEnumerator.get_possible_plans(
                    query_node, state, join_graph,
                    planner_config=None,
                    avoid_eq_filters=False,
                    workload_info=self.workload_info,
                    use_plan_restrictions=self.use_plan_restrictions
                )
                _, valid_new_states = self._make_new_states(
                    state, [0.0] * len(possible_plans), possible_plans)
                rand_idx = np.random.randint(len(valid_new_states))
                state = valid_new_states[rand_idx]
            predicted = self.infer(query_node, [state[0]])
            return predicted, state

        best_predicted = [np.inf]
        best_state = None
        for _ in range(num_random_plans):
            state = query_leaves
            assert len(state) == num_rels, len(state)
            predicted, state = _SampleOne(state)
            if predicted[0] < best_predicted[0]:
                best_predicted = predicted
                best_state = state

        planning_time = (time.time() - planning_start_t) * 1e3
        predicted = best_predicted
        state = best_state
        print('Found best random plan out of {}:'.format(num_random_plans))
        print('  {:.1f} {}'.format(predicted[0], str([s.hint_str(self.plan_physical) for s in state])))
        print('Planning took {:.1f}ms'.format(planning_time))
        return predicted[0], state[0]
