import collections
import copy
import gc
import glob
import os
import pickle
import pprint
import time
from parser import plan_node
from typing import Any, Dict, List, Tuple

import numpy as np
from common import workload
from db.pg_conn import PostgresConnector
from expert_pool.plan_gen_expert.dataset.expert_datasets import SubPlanTrainingPoint

# Import from local package
from expert_pool.plan_gen_expert.encoders import (
    plan_graph_encoder,
)
from expert_pool.plan_gen_expert.encoders import (
    sql_graph_encoder as query_graph_encoder,
)


class Experience:
    """
    A class representing an experience replay buffer for query optimization.
    It manages nodes from workloads, featurizes them, and handles saving/loading.
    """

    def __init__(
        self,
        db_cli: PostgresConnector,
        data: List[workload.Node],
        workload_info: workload.WorkloadInfo = None,
        plan_featurizer_cls=None,
        query_featurizer_cls=None,
        query_featurizer=None,
        use_tree_conv=True,
    ):
        self.db_cli = db_cli
        self.workload_info = workload_info
        self.use_tree_conv = use_tree_conv
        self.plan_featurizer_cls = plan_featurizer_cls
        self.query_featurizer_cls = query_featurizer_cls

        assert isinstance(query_featurizer, query_graph_encoder.SimQueryFeaturizer)
        self.query_featurizer = query_featurizer
        self.plan_featurizer = None
        self.pos_featurizer = None

        # Process and initialize nodes
        self._initialize_nodes(data)

    def _initialize_nodes(self, data: List[workload.Node]) -> None:
        """Initializes the nodes by filtering, gathering info, and estimating selectivity."""

        # by default, use this
        self.nodes = [ele.filter_scans_joins() for ele in data]
        self.initial_size = len(self.nodes)

        # Gather filter info and estimate selectivity for all nodes
        for node in self.nodes:
            node.gather_filter_info()
        plan_node.PGToNodeHelper.estimate_nodes_selectivility(self.db_cli, self.nodes)

    def save(self, path: str) -> None:
        """Saves all Nodes in the current replay buffer to a file using pickle."""
        if os.path.exists(path):
            old_path = path
            path = "{}-{}".format(old_path, time.time())
            print("Path {} exists, appending current time: {}".format(old_path, path))
            assert not os.path.exists(path), path
        to_save = (self.initial_size, self.nodes)
        with open(path, "wb") as f:
            pickle.dump(to_save, f)
        print("Saved Experience to:", path)

    def load(
        self,
        path_glob: str,
        keep_last_fraction: float = 1.0,
    ) -> None:
        """Loads multiple serialized Experience buffers into a single one.
        The 'initial_size' Nodes from self would be kept, while those from the
        loaded buffers would be dropped. Internally, checked that all buffers
        and self have the same 'initial_size' field.
        """
        paths = glob.glob(os.path.expanduser(path_glob))
        if not paths:
            raise ValueError("No replay buffer files found")
        assert 0 <= keep_last_fraction <= 1, keep_last_fraction

        # query name -> set(plan string)
        total_unique_plans_table = collections.defaultdict(set)
        total_num_unique_plans = 0
        initial_nodes_len = len(self.nodes)
        for path in paths:
            t1 = time.time()
            print("Loading replay buffer", path)
            gc.disable()
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            gc.enable()
            print(" ...took {:.1f} seconds".format(time.time() - t1))
            initial_size, nodes = loaded
            # Sanity checks
            assert type(initial_size) is int and type(nodes) is list, path
            assert initial_size == self.initial_size, (
                path,
                initial_size,
                self.initial_size,
            )
            assert len(nodes) >= initial_size and len(nodes) % initial_size == 0, (
                len(nodes),
                path,
            )

            nodes_executed = nodes[initial_size:]
            if keep_last_fraction < 1:
                assert len(nodes_executed) % initial_size == 0
                num_iters = len(nodes_executed) // initial_size
                keep_num_iters = int(num_iters * keep_last_fraction)
                print(
                    " orig len {} keeping the last fraction {} ({} iters)".format(
                        len(nodes_executed), keep_last_fraction, keep_num_iters
                    )
                )
                nodes_executed = nodes_executed[-(keep_num_iters * initial_size) :]
            self.nodes.extend(nodes_executed)

            # Analysis of unique plans
            num_unique_plans, unique_plans_table = self._count_unique_plans(
                self.initial_size, nodes_executed
            )
            total_num_unique_plans_prev = total_num_unique_plans
            total_num_unique_plans = self._merge_unique_plans_into(
                unique_plans_table, total_unique_plans_table
            )
            print(
                " num_unique_plans from loaded buffer {}; actually new unique plans contributed (after merging) {}".format(
                    num_unique_plans,
                    total_num_unique_plans - total_num_unique_plans_prev,
                )
            )

        print(
            "Loaded {} nodes from {} buffers; glob={}, paths:\n{}".format(
                len(self.nodes) - initial_nodes_len,
                len(paths),
                path_glob,
                "\n".join(paths),
            )
        )
        print("Total unique plans (num_query_execs):", total_num_unique_plans)

    @staticmethod
    def _count_unique_plans(
        num_templates: int, nodes: List[workload.Node]
    ) -> Tuple[int, Dict[str, set]]:
        """Counts unique plans for the given nodes."""
        assert len(nodes) % num_templates == 0, (len(nodes), num_templates)
        unique_plans = collections.defaultdict(set)
        for i in range(num_templates):
            query_name = nodes[i].info["query_name"]
            hint_set = unique_plans[query_name]
            for j, node in enumerate(nodes[i::num_templates]):
                assert node.info["query_name"] == query_name, (
                    node.info["query_name"],
                    query_name,
                )
                hint = node.hint_str(with_physical_hints=True)
                hint_set.add(hint)
        num_unique_plans = sum([len(s) for s in unique_plans.values()])
        return num_unique_plans, unique_plans

    @staticmethod
    def _merge_unique_plans_into(
        from_table: Dict[str, set], into_table: Dict[str, set]
    ) -> int:
        """Merges unique plans from one table into another and returns the total count."""
        for query_name, unique_plans in from_table.items():
            into_table[query_name] = into_table[query_name].union(unique_plans)
        return sum([len(s) for s in into_table.values()])

    def prepare(self, rewrite_generic: bool = False) -> None:
        """Prepares the workload information and featurizers."""
        if rewrite_generic:
            print("Rewriting all joins -> Join, all scans -> Scan")
            workload.NodeOps.rewrite_as_generic_joinscan(self.nodes)

        rel_names, scan_types, join_types, all_ops = (
            self.workload_info.rel_names,
            self.workload_info.scan_types,
            self.workload_info.join_types,
            self.workload_info.all_ops,
        )
        print("{} rels: {}".format(len(rel_names), rel_names))
        print(
            "{} rel_ids: {}".format(
                len(self.workload_info.rel_ids), self.workload_info.rel_ids
            )
        )
        print("{} scans: {}".format(len(scan_types), scan_types))
        print("{} joins: {}".format(len(join_types), join_types))
        print("{} all ops: {}".format(len(all_ops), all_ops))

        if self.use_tree_conv:
            assert issubclass(
                self.plan_featurizer_cls, plan_graph_encoder.PhysicalTreeNodeFeaturizer
            )
            self.plan_featurizer = self.plan_featurizer_cls(self.workload_info)
            self.pos_featurizer = None
        else:
            self.plan_featurizer = self.plan_featurizer_cls(self.workload_info)
            self.pos_featurizer = plan_graph_encoder.ParentPositionFeaturizer(
                self.workload_info
            )

        if self.query_featurizer is None:
            self.query_featurizer = self.query_featurizer_cls(self.workload_info)

        if isinstance(
            self.plan_featurizer, plan_graph_encoder.PreOrderSequenceFeaturizer
        ):
            ns = workload.NodeOps.get_all_subtrees([self.nodes[0]])
            f = self.plan_featurizer(ns[0])
            pf = self.pos_featurizer(ns[0])
            assert len(f) == len(pf), (len(f), len(pf))

    def get_first_index_for_template(
        self,
        template_index: int,
        skip_first_n: int,
        use_last_n_iters: int = -1,
    ) -> int:
        """Calculates the starting index for a template, allowing featurizing just last n iters' data."""
        assert skip_first_n in [0, self.initial_size], (skip_first_n, self.initial_size)
        num_iters = (len(self.nodes) - skip_first_n) // self.initial_size
        i_start = template_index + skip_first_n
        if use_last_n_iters < 0 or num_iters <= use_last_n_iters:
            return i_start
        return i_start + (num_iters - use_last_n_iters) * self.initial_size

    def compute_best_latencies(
        self,
        template_index: int,
        skip_first_n: int,
        with_physical_hints: bool,
        skip_training_on_timeouts: bool,
    ) -> Tuple[Dict[str, Tuple[float, workload.Node]], int, List[List[workload.Node]]]:
        """Computes best latencies per subplan for a particular query template."""
        i_start = template_index + skip_first_n
        subplan_to_best = {}  # Hint str -> (cost, subplan)
        num_subtrees = 0
        all_subtrees = []
        for j, node in enumerate(self.nodes[i_start :: self.initial_size]):
            if skip_training_on_timeouts and getattr(node, "is_timeout", False):
                all_subtrees.append([])
                continue
            subtrees = workload.NodeOps.get_all_subtrees_no_leaves(node)
            num_subtrees += len(subtrees)
            all_subtrees.append(subtrees)
            for t in subtrees:
                t_key = t.hint_str(with_physical_hints)
                curr_cost, _ = subplan_to_best.get(t_key, (1e30, None))
                if node.cost < curr_cost:
                    subplan_to_best[t_key] = (node.cost, t)
        return subplan_to_best, num_subtrees, all_subtrees

    def _featurize_dedup(
        self,
        rewrite_generic: bool = False,
        skip_first_n: int = 0,
        physical_execution_hindsight: bool = False,
        on_policy: bool = False,
        use_last_n_iters: int = -1,
        use_new_data_only: bool = False,
        skip_training_on_timeouts: bool = False,
    ) -> Tuple[List, List, List, List, int]:
        """Featurizes with deduplication logic."""
        if use_last_n_iters > 0 and use_new_data_only:
            print(
                "Both use_last_n_iters > 0 and use_new_data_only are set: letting the latter take precedence."
            )
        if on_policy:
            assert (
                use_last_n_iters == -1
            ), "Cannot have both on_policy and use_last_n_iters={} set.".format(
                use_last_n_iters
            )
            use_last_n_iters = 1
        self.prepare(rewrite_generic)

        all_query_vecs = []
        all_feat_vecs = []
        all_pos_vecs = []
        all_costs = []
        all_subtrees = []
        num_total_subtrees = 0
        num_new_datapoints = 0

        for i in range(self.initial_size):
            subplan_to_best, num_subtrees, subtrees = self.compute_best_latencies(
                i,
                skip_first_n,
                with_physical_hints=not rewrite_generic,
                skip_training_on_timeouts=skip_training_on_timeouts,
            )
            num_total_subtrees += num_subtrees

            to_featurize = []
            if use_last_n_iters > 0:
                for iter_k_subplans in subtrees[-use_last_n_iters:]:
                    for subplan in iter_k_subplans:
                        key = subplan.hint_str(with_physical_hints=not rewrite_generic)
                        best_cost, _ = subplan_to_best[key]
                        to_featurize.append((best_cost, subplan))
                if subtrees:
                    last_iter_subplans = subtrees[-1]
                    last_iter_template_cost = self.nodes[-self.initial_size + i].cost
                    new_data_points = []
                    for tup in to_featurize[-len(last_iter_subplans) :]:
                        if tup[0] == last_iter_template_cost:
                            num_new_datapoints += 1
                            new_data_points.append(tup)
                    if use_new_data_only:
                        to_featurize = new_data_points
            else:
                to_featurize = subplan_to_best.values()

            query_feat = self.query_featurizer(self.nodes[i + skip_first_n])
            all_query_vecs.extend([query_feat] * len(to_featurize))

            if not self.use_tree_conv:
                for best_cost, best_subplan in to_featurize:
                    all_costs.append(best_cost)
                    all_feat_vecs.append(self.plan_featurizer(best_subplan))
                    all_pos_vecs.append(self.pos_featurizer(best_subplan))
            else:
                for best_cost, best_subplan in to_featurize:
                    all_costs.append(best_cost)
                    all_subtrees.append(best_subplan)

        if self.use_tree_conv and all_subtrees:
            all_feat_vecs, all_pos_vecs = plan_graph_encoder.make_and_featurize_trees(
                self.plan_featurizer, all_subtrees
            )

        return (
            all_query_vecs,
            all_feat_vecs,
            all_pos_vecs,
            all_costs,
            num_new_datapoints,
        )

    def _featurize_hindsight_relabeling(
        self,
        rewrite_generic: bool = False,
        verbose: bool = False,
        skip_first_n: int = 0,
        physical_execution_hindsight: bool = False,
        use_last_n_iters: int = -1,
    ) -> Tuple[List, List, List, List]:
        """Featurizes with hindsight relabeling logic."""
        assert self.use_tree_conv
        self.prepare(rewrite_generic)

        all_query_vecs = []
        all_feat_vecs = []
        all_pos_vecs = []
        all_costs = []
        all_subtrees = []
        num_total_subtrees = 0

        def _top_down_collect(
            node: workload.Node,
            hindsight_goal_costs: List[Tuple[workload.Node, float]],
            accum: List[SubPlanTrainingPoint],
            info_to_attach: Dict[str, Any],
            is_top_level: bool = True,
        ) -> None:
            assert node.IsJoin() or node.IsScan(), node
            if node.IsScan():
                return
            if node.actual_time_ms is not None or (
                skip_first_n == 0
                and len(self.nodes) == self.initial_size
                and is_top_level
            ):
                goal = node
                goal.info = copy.deepcopy(info_to_attach)
                goal_cost = (
                    node.cost if node.actual_time_ms is None else node.actual_time_ms
                )
                hindsight_goal_costs.append((goal, goal_cost))
            for goal, goal_cost in hindsight_goal_costs:
                accum.append(
                    SubPlanTrainingPoint(subplan=node, goal=goal, cost=goal_cost)
                )
            for c in node.children:
                _top_down_collect(
                    c, hindsight_goal_costs, accum, info_to_attach, is_top_level=False
                )

        for i in range(self.initial_size):
            i_start = self.get_first_index_for_template(
                i, skip_first_n, use_last_n_iters
            )
            accum = []
            info_to_attach = {
                k: self.nodes[i].info[k] for k in ["all_filters_est_rows"]
            }
            for node in self.nodes[i_start :: self.initial_size]:
                _top_down_collect(node.Copy(), [], accum, info_to_attach)
            num_total_subtrees += len(accum)

            best = collections.defaultdict(lambda: np.inf)
            ret = {}
            for point in accum:
                key = point.ToSubplanGoalHint(with_physical_hints=not rewrite_generic)
                if point.cost < best[key]:
                    best[key] = point.cost
                    ret[key] = point

            for point in ret.values():
                all_query_vecs.append(self.query_featurizer(point.goal))
                all_costs.append(point.cost)
                all_subtrees.append(point.subplan)

            if verbose:
                print(
                    "{} subplan_to_best, {} entries".format(
                        self.nodes[i].info["query_name"], len(ret)
                    )
                )
                pprint.pprint(list(ret.values()))

        all_feat_vecs, all_pos_vecs = plan_graph_encoder.make_and_featurize_trees(
            self.plan_featurizer, all_subtrees
        )
        print(
            "num_total_subtrees={} num_unique_subtrees={}".format(
                num_total_subtrees, len(all_query_vecs)
            )
        )
        return all_query_vecs, all_feat_vecs, all_pos_vecs, all_costs

    def featurize(
        self,
        rewrite_generic: bool = False,
        verbose: bool = False,
        skip_first_n: int = 0,
        deduplicate: bool = False,
        physical_execution_hindsight: bool = False,
        on_policy: bool = False,
        use_last_n_iters: int = -1,
        use_new_data_only: bool = False,
        skip_training_on_timeouts: bool = False,
    ) -> Tuple[List, List, List, List]:
        """Main featurization entry point. Dispatches to appropriate featurization logic based on flags."""
        if physical_execution_hindsight:
            assert deduplicate, "physical_execution_hindsight requires deduplicate"
            return self._featurize_hindsight_relabeling(
                rewrite_generic=rewrite_generic,
                verbose=verbose,
                skip_first_n=skip_first_n,
                physical_execution_hindsight=physical_execution_hindsight,
                use_last_n_iters=use_last_n_iters,
            )
        if use_last_n_iters == -1 and not deduplicate:
            assert skip_first_n in [0, self.initial_size], (
                skip_first_n,
                self.initial_size,
            )
            num_iters = (len(self.nodes) - skip_first_n) // self.initial_size
            use_last_n_iters = num_iters  # Use all iterations if -1

        return self._featurize_dedup(
            rewrite_generic=rewrite_generic,
            skip_first_n=skip_first_n,
            physical_execution_hindsight=physical_execution_hindsight,
            on_policy=on_policy,
            use_last_n_iters=use_last_n_iters,
            use_new_data_only=use_new_data_only,
            skip_training_on_timeouts=skip_training_on_timeouts,
        )[
            :-1
        ]  # Exclude num_new_datapoints from return

    def add(self, node: workload.Node) -> None:
        """Adds a new node to the buffer."""
        self.nodes.append(node)

    def drop_agent_experience(self) -> None:
        """Drops agent experience, keeping only initial nodes."""
        old_len = len(self.nodes)
        self.nodes = self.nodes[: self.initial_size]
        print(
            "Dropped agent experience (prev len {}, new len {})".format(
                old_len, len(self.nodes)
            )
        )
