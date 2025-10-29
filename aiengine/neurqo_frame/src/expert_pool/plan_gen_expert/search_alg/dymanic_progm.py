import collections
import copy
import numpy as np
from common import hyperparams
from typing import List, Dict, Tuple, DefaultDict
import networkx as nx
from parser import sql_parser
from db.pg_conn import PostgresConnector
from common.workload import Node, NodeOps

# Import from local package
from expert_pool.plan_gen_expert.models import cost_model


class DynamicProgramming:
    """Implements bottom-up dynamic programming for query plan optimization."""

    @classmethod
    def Params(cls):
        """
        Defines and returns the parameters for the DynamicProgramming class.

        Returns:
            hyperparams.InstantiableParams: Configured parameters for the class.
        """
        p = hyperparams.InstantiableParams(cls)
        p.define('cost_model', cost_model.NullCost.Params(), 'Parameters for the cost model.')
        p.define('search_space', 'bushy', 'Search space options: bushy, dbmsx, bushy_norestrict.')
        p.define('plan_physical_ops', False, 'Whether to plan physical join and scan operations.')
        p.define('collect_data_include_suboptimal', True, 'Whether to call enumeration hooks on suboptimal plans.')
        return p

    def __init__(self, params, cursor: PostgresConnector):
        """
        Initializes the DynamicProgramming planner with given parameters.

        Args:
            params: Configuration parameters for the planner.
        """
        self.params = params.copy()
        p = self.params
        self.cost_model = p.cost_model.cls(p.cost_model, cursor)
        self.on_enumerated_hooks = []
        assert p.search_space in ('bushy', 'dbmsx', 'bushy_norestrict'), 'Invalid search space.'
        # those are only for tesitng
        self.join_ops = ['Join']
        self.scan_ops = ['Scan']
        self.use_plan_restrictions = (p.search_space != 'bushy_norestrict')

    def set_physical_ops(self, join_ops, scan_ops):
        """
        Sets physical join and scan operations if planning physical ops is enabled.

        Args:
            join_ops (list): List of join operations to use.
            scan_ops (list): List of scan operations to use.
        """
        p = self.params
        assert p.plan_physical_ops, 'Physical ops planning must be enabled.'
        self.join_ops = copy.deepcopy(join_ops)
        self.scan_ops = copy.deepcopy(scan_ops)

    def push_on_enumerated_hook(self, func):
        """
        Adds a hook function to be called on each enumerated and costed subplan.

        Args:
            func: Function to execute on each subplan (Node, cost).
        """
        self.on_enumerated_hooks.append(func)

    def pop_on_enumerated_hook(self):
        """
        Removes the most recently added enumeration hook.

        Original function name: PopOnEnumeratedHook
        """
        self.on_enumerated_hooks.pop()

    def collect_for_single_query(self, query_node: Node):
        """
        Executes dynamic programming to optimize a query plan.

        Args:
            query_node: Node representing the query.

        Returns:
            tuple: (best_node, dp_tables)
                - best_node (Node): The optimal query plan.
                - dp_tables (dict): Dictionary mapping relation set sizes to plans and costs.
        """
        join_graph, all_join_conds = query_node.get_parse_update_sql(sql_parse_func=sql_parser.simple_parse_sql)
        query_leaves = query_node.CopyLeaves()

        dp_tables = collections.defaultdict(dict)
        for leaf_node in query_leaves:
            dp_tables[1][leaf_node.table_alias] = (0, leaf_node)
        return self._dp_bushy_search_space(query_node, join_graph, all_join_conds, query_leaves, dp_tables)

    def _dp_bushy_search_space(
            self,
            original_node: Node,
            join_graph: nx.Graph,
            all_join_conds: List[str],
            query_leaves: List[Node],
            dp_tables: DefaultDict[int, Dict[str, Tuple[float, Node]]]
    ) -> Tuple[Node, DefaultDict[int, Dict[str, Tuple[float, Node]]], int]:
        """
        Implements bushy search space for dynamic programming.

        Args:
            original_node: Original query node.
            join_graph: Graph of join relationships.
            all_join_conds: List of join conditions.
            query_leaves: List of leaf nodes (base tables).
            dp_tables: Dictionary to store dynamic programming tables.

        Returns:
            tuple: (best_node, dp_tables) for the optimal plan and tables.
        """
        _cost_model_num = 0
        p = self.params
        num_rels = len(query_leaves)
        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]

                for l_ids, l_tup in dp_table_i.items():
                    for r_ids, r_tup in dp_table_j.items():
                        l: Node = l_tup[1]  # 0 is cost, 1 is the node
                        r: Node = r_tup[1]
                        if not NodeOps.exists_join_edge_in_graph(l, r, join_graph):
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))

                        for join in NodeOps.enumerate_join_with_ops(
                                l, r, self.join_ops, self.scan_ops, use_plan_restrictions=self.use_plan_restrictions):
                            join_conds = join.KeepRelevantJoins(all_join_conds)
                            cost = self.cost_model(join, join_conds)

                            _cost_model_num += 1

                            if p.collect_data_include_suboptimal:
                                for hook in self.on_enumerated_hooks:
                                    hook(join, cost)
                            if join_ids not in dp_table or dp_table[join_ids][0] > cost:
                                dp_table[join_ids] = (cost, join)

        if not p.collect_data_include_suboptimal:
            for level in range(2, num_rels + 1):
                dp_table = dp_tables[level]
                for ids, tup in dp_table.items():
                    cost, plan = tup[0], tup[1]
                    for hook in self.on_enumerated_hooks:
                        hook(plan, cost)
        return list(dp_tables[num_rels].values())[0][1], dp_tables, _cost_model_num


if __name__ == '__main__':
    p = DynamicProgramming.Params()
    print(p)
    dp = p.cls(p)
    print(dp)
    node = Node('Scan', table_name='title').with_alias('t')
    node.info['sql_str'] = 'SELECT * FROM title t;'
    print(dp.run(node))
