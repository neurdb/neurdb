# Standard library imports
import collections
import random
from typing import Dict, List, Tuple

# Third-party imports
import pandas as pd

# Local/project imports
from common import workload
from db.pg_conn import PostgresConnector
from parser import sql_parser


class PGToNodeHelper:

    @staticmethod
    def _fuse_hints(sql: str, comment: str, explain_str: str = 'explain(verbose, format json)'):
        end_of_comment_idx = sql.find('*/')
        if end_of_comment_idx == -1:
            existing_comment = None
        else:
            split_idx = end_of_comment_idx + len('*/\n')
            existing_comment = sql[:split_idx]
            sql = sql[split_idx:]

        # Fuse hint comments.
        if comment:
            assert comment.startswith('/*+') and comment.endswith('*/'), (
                'Don\'t know what to do with these', sql, existing_comment, comment)
            if existing_comment is None:
                fused_comment = comment
            else:
                comment_body = comment[len('/*+ '):-len(' */')].rstrip()
                existing_comment_body_and_tail = existing_comment[len('/*+'):]
                fused_comment = '/*+\n' + comment_body + '\n' + existing_comment_body_and_tail
        else:
            fused_comment = existing_comment

        if fused_comment:
            s = fused_comment + '\n' + str(explain_str).rstrip() + '\n' + sql
        else:
            s = str(explain_str).rstrip() + '\n' + sql

        return s

    @staticmethod
    def sql_to_plan_node(cursor: PostgresConnector,
                         sql: str,
                         comment: str = None,
                         keep_scans_joins_only=False,
                         ) -> Tuple[workload.Node, Dict]:
        """Issues EXPLAIN (FORMAT JSON) on a SQL string; parse into our AST node.

        Args:
            sql: SQL query string.
            comment: Optional hint comment (/*+ ... */) or SET statements.
            keep_scans_joins_only: If True, filter out non-scan/join operators.
            cursor: Optional PostgresConnector (already connected).

        Returns:
            (node, json_dict): parsed plan node and raw plan JSON.
        """
        if cursor is None:
            raise ValueError("sql_to_plan_node requires a PostgresConnector (cursor)")

        # Use PostgresConnector.explain_read_sql directly
        new_sql = PGToNodeHelper._fuse_hints(sql=sql, comment=comment, explain_str='explain(verbose, format json)')
        geqo_off = "off" if comment is not None and len(comment) > 0 else "on"
        plan_json = cursor.explain_read_sql(new_sql, geqo_off)

        node = workload.Node.plan_json_to_node(plan_json)
        if keep_scans_joins_only:
            return node.filter_scans_joins(), plan_json
        return node, plan_json

    @staticmethod
    def sql_to_plan_node_w_parsed_sql(db_cli: PostgresConnector, sql_str, sql_file_name):
        # node is the instance of the Node in Plan Node
        # todo. this should get from the buffer. sqllite.
        node, json_dict = PGToNodeHelper.sql_to_plan_node(db_cli, sql_str)
        node.info['path'] = sql_file_name
        node.info['sql_str'] = sql_str
        node.info['query_name'] = sql_file_name
        node.info['explain_json'] = json_dict
        graph, join_conds = node.get_parse_update_sql(sql_parse_func=sql_parser.simple_parse_sql)
        node.gather_filter_info()
        return node

    @staticmethod
    def estimate_nodes_selectivility(db_cli: PostgresConnector, nodes: List[workload.Node]):
        """ EstimateFilterRows
        For each node, issues an EXPLAIN to estimate #rows of unary predicates.

        Writes result back into node.info['all_filters_est_rows'],
        as { relation_id: num_rows }.
        """

        cache = {}
        for node in nodes:
            for table_id, pred in node.info['all_filters'].items():
                key = (table_id, pred)
                if key not in cache:
                    sql = f"EXPLAIN(format json) SELECT * FROM {table_id} WHERE {pred};"
                    try:
                        plan_json = db_cli.explain_read_sql(sql, "default")  # 返回 dict
                    except Exception as e:
                        print(f"[estimate_nodes_selectivility]: cannot explian sql: {e}, use full table rows estimates")
                        # cannot parse the subsql, then we use the full table
                        sql = f"EXPLAIN(format json) SELECT * FROM {table_id};"
                        plan_json = db_cli.explain_read_sql(sql, "default")

                    num_rows = plan_json['Plan']['Plan Rows']
                    cache[key] = num_rows
        print("{} unique filters, with example <(table_id, filter): number rows> = {}".format(
            len(cache), random.choice(list(cache.items())) if cache else None))
        for node in nodes:
            d = {}
            for table_id, pred in node.info['all_filters'].items():
                d[table_id] = cache[(table_id, pred)]
            node.info['all_filters_est_rows'] = d

    @staticmethod
    def get_real_pg_latency(db_cli: PostgresConnector, sql: str, hint: str):
        # GEQO must be disabled for hinting larger joins to work.
        # Why 'verbose': makes plan_json_to_node() able to access required
        # fields, e.g., 'Output' and 'Alias'.  Also see sql_to_plan_node() comment.
        geqo_off = hint is not None and len(hint) > 0
        if geqo_off:
            db_cli.set_geqo_exp("off")

        s = PGToNodeHelper._fuse_hints(sql=sql, comment=hint, explain_str="")
        result, _ = db_cli.explain_analysis_read_sql(s)

        json_dict = result[0][0][0]
        latency = float(json_dict['Execution Time'])
        node = workload.Node.plan_json_to_node(json_dict)
        new_node = node.filter_scans_joins()

        # if check_hint_used:
        #     expected = hint
        #     actual = node.hint_str(with_physical_hints=cost_model.ContainsPhysicalHints(hint))
        #     assert expected == actual, 'Expected={}\nActual={}, actual node:\n{}\nSQL=\n{}'.format(
        #         expected, actual, node, sql)

        return latency


class Stats:
    """Plan statistics."""

    def __init__(self):
        self.join_counts = collections.defaultdict(int)
        self.scan_counts = collections.defaultdict(int)
        self.shape_counts = collections.defaultdict(int)
        self.nested_loop_children_counts = collections.defaultdict(int)
        self.num_plans = 0

    def Update(self, nodes):
        for node in nodes:
            join_ops, scan_ops, nl_children = self.GetOps(node)
            for op in join_ops:
                self.join_counts[op] += 1
            for op in scan_ops:
                self.scan_counts[op] += 1
            for nl_children_ops in nl_children:
                self.nested_loop_children_counts[nl_children_ops] += 1

            shape = self.GetShape(node)
            self.shape_counts[shape] += 1
        self.num_plans += len(nodes)
        return self

    def GetShape(self, node):

        def IsLeftDeep(n):
            if n.IsScan():
                return True
            if n.children[1].IsJoin():
                return False
            return IsLeftDeep(n.children[0])

        def IsRightDeep(n):
            if n.IsScan():
                return True
            if n.children[0].IsJoin():
                return False
            return IsRightDeep(n.children[1])

        if IsLeftDeep(node):
            shape = 'left_deep'
        elif IsRightDeep(node):
            shape = 'right_deep'
        else:
            shape = 'bushy'
        return shape

    def GetOps(self, node):
        join_ops, scan_ops, nl_children = [], [], []

        def Fn(n):
            if n.IsJoin():
                join_ops.append(n.node_type)
                if n.node_type == 'Nested Loop':
                    nl_children.append(
                        (n.children[0].node_type, n.children[1].node_type))
            elif n.IsScan():
                scan_ops.append(n.node_type)

        workload.NodeOps.map_node(node, Fn)
        return join_ops, scan_ops, nl_children

    def Print(self):
        print('Total num plans:', self.num_plans)

        def DoPrint(cnts):
            df = pd.DataFrame(cnts,
                              index=['count']).T.sort_values('count',
                                                             ascending=False)
            df['%'] = df['count'] / df['count'].sum() * 100.0
            df['%'] = df['%'].apply(lambda t: '{:.0f}%'.format(t))
            df['count'] = df['count'].apply(self.HumanFormat)
            print(df)
            print()

        print('Join ops')
        DoPrint(self.join_counts)
        print('Scan ops')
        DoPrint(self.scan_counts)
        print('Shapes')
        DoPrint(self.shape_counts)

        print('NL children')
        DoPrint(self.nested_loop_children_counts)

    @staticmethod
    def HumanFormat(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


class PlanAnalysis:
    """Plan analysis of a list of Nodes."""

    def __init__(self):
        self.total_stats = Stats()

    @classmethod
    def Build(cls, nodes):
        analysis = PlanAnalysis()
        return analysis.Update(nodes)

    def Update(self, nodes):
        self.total_stats.Update(nodes)
        return self

    def Print(self):
        print('===== Plan Analysis =====')
        self.total_stats.Print()
