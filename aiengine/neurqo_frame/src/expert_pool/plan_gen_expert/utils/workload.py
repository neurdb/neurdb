import collections
import copy
import functools
import re
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
from common import BaseConfig


class Node:
    """Basic AST node class.

    Example usage:
       n = Node('Nested Loop')
       n.cost = 23968.1
       n.info = {'explain_json': json_dict, 'sql_str': sql_str, ...}
       n.children = [...]
    """

    @classmethod
    def plan_json_to_node(cls, json_dict: Dict):
        """Takes JSON dict, parses into a Node."""
        curr = json_dict["Plan"]

        def _parse_pg(json_dict, select_exprs=None, indent=0):
            op = json_dict["Node Type"]
            cost = json_dict["Total Cost"]
            if op == "Aggregate":
                op = json_dict["Partial Mode"] + op
                if select_exprs is None:
                    # Record the SELECT <select_exprs> at the topmost Aggregate.
                    # E.g., ['min(mi.info)', 'min(miidx.info)', 'min(t.title)'].
                    select_exprs = json_dict["Output"]

            # Record relevant info.
            curr_node = cls(op)
            curr_node.cost = cost
            # Only available if 'analyze' is set (actual execution).
            curr_node.actual_time_ms = json_dict.get("Actual Total Time")
            if "Relation Name" in json_dict:
                curr_node.table_name = json_dict["Relation Name"]
                curr_node.table_alias = json_dict["Alias"]

            # Unary predicate on a table.
            if "Filter" in json_dict:
                assert "Scan" in op, json_dict
                assert "Relation Name" in json_dict, json_dict
                curr_node.info["filter"] = json_dict["Filter"]

            if "Scan" in op and select_exprs:
                # Record select exprs that belong to this leaf.
                # Assume: SELECT <exprs> are all expressed in terms of aliases.
                filtered = NodeOps.filter_exprs_by_alias(
                    select_exprs, json_dict["Alias"]
                )
                if filtered:
                    curr_node.info["select_exprs"] = filtered

            # Recurse.
            if "Plans" in json_dict:
                for next_plan in json_dict["Plans"]:
                    # Treating Bitmap Heap Scans as the leaf node, since there is always just
                    # a single Bitmap Index Scan node below
                    if (curr_node.node_type != "Bitmap Heap Scan") or (
                        next_plan["Node Type"] != "Bitmap Index Scan"
                    ):
                        curr_node.children.append(
                            _parse_pg(
                                next_plan, select_exprs=select_exprs, indent=indent + 2
                            )
                        )

            # Special case.
            if op == "Bitmap Heap Scan":
                for c in curr_node.children:
                    if c.node_type == "Bitmap Index Scan":
                        # 'Bitmap Index Scan' doesn't have the field 'Relation Name'.
                        c.table_name = curr_node.table_name
                        c.table_alias = curr_node.table_alias

            return curr_node

        return _parse_pg(curr)

    def __init__(self, node_type, table_name=None, cost=None):
        self.node_type = node_type
        self.cost = cost  # Total cost.
        self.actual_time_ms = None

        # fill the info with the needed fields
        self.info = {
            # 'sql_str': None,
            # 'path': None,
            # 'query_name': None,
            # 'explain_json': None,
            # 'parsed_join_graph': None,
            # 'parsed_join_conds': None,
            # 'overall_join_graph': None,
            # 'overall_join_conds': None,
            # 'select_exprs': [],
            # 'filter': None,
            # 'eq_filters': [],
            # 'filtered_attributes': [],
            # 'all_filters': {}
        }

        self.children = []

        # Set for leaves (i.e., scan nodes).
        self.table_name = table_name
        self.table_alias = None

        # Internal cached fields.
        self._card = None  # Used in MinCardCost.
        self._leaf_scan_op_copies = {}

    def with_alias(self, alias):
        self.table_alias = alias
        return self

    def get_table_id(self, with_alias=True, alias_only=False):
        """Table id for disambiguation."""
        if with_alias and self.table_alias:
            if alias_only:
                return self.table_alias
            return self.table_name + " AS " + self.table_alias
        assert self.table_name is not None
        return self.table_name

    @functools.lru_cache(maxsize=128)
    def to_str(self, with_cost=True, indent=0):
        s = "" if indent == 0 else " " * indent
        if self.table_name is None:
            if with_cost:
                s += "{} cost={}\n".format(self.node_type, self.cost)
            else:
                s += "{}\n".format(self.node_type)
        else:
            if with_cost:
                s += "{} [{}] cost={}\n".format(
                    self.node_type, self.get_table_id(), self.cost
                )
            else:
                s += "{} [{}]\n".format(
                    self.node_type,
                    self.get_table_id(),
                )
        for c in self.children:
            s += c.to_str(with_cost=with_cost, indent=indent + 2)
        return s

    def get_parse_update_sql(
        self, sql_parse_func: Callable[[str], Dict[str, Any]]
    ) -> Tuple[nx.MultiGraph, List]:
        """Parses the join graph of this node into (nx.Graph, join conds).

        If self.info['sql_str'] exists, parses this SQL string. Otherwise
        parses the result of self.to_sql(self.info['all_join_conds'])---this is
        usually used for manually constructued sub-plans.

        Internally, try to read from a cached pickle file if it exists.
        """
        graph = self.info.get("parsed_join_graph")
        join_conds = self.info.get("parsed_join_conds")
        if graph is None or join_conds is None:
            sql_str = self.info.get("sql_str") or self.to_sql(
                self.info["overall_join_conds"], with_filters=False
            )

            graph, join_conds = sql_parse_func(sql_str)

            assert graph is not None, sql_str
            assert len(graph.edges) == len(
                join_conds
            ), "Join graph and conditions mismatch."

            self.info["parsed_join_graph"] = graph
            self.info["parsed_join_conds"] = join_conds
        return graph, join_conds

    def GetOrParseJoinGraph(self):
        """Returns this Node's join graph as a networkx.Graph."""
        graph = self.info.get("parsed_join_graph")
        if graph is None:
            # Restricting an overall graph onto this Node's leaf IDs.
            overall_graph = self.info.get("overall_join_graph")
            if overall_graph is not None:
                subgraph_view = overall_graph.subgraph(self.leaf_ids(alias_only=True))
                assert len(subgraph_view) > 0
                graph = subgraph_view
                self.info["parsed_join_graph"] = subgraph_view
                return graph
            return self.get_parse_update_sql()[0]
        return graph

    def GetSelectExprs(self):
        """Returns a list of SELECT exprs associated with this Node.

        These expressions are the ultimate query outputs.  During parsing into
        balsa.Node objects, we push down that original list to the corresponding
        leaves.
        """
        select_exprs = []

        def _Fn(l):
            exprs = l.info.get("select_exprs")
            if exprs:
                select_exprs.extend(exprs)

        NodeOps.map_leaves(self, _Fn)
        return select_exprs

    def GetFilters(self):
        """Returns a list of filter conditions associated with this Node.

        The filters are parsed by Postgres and for each table, include all
        pushed-down predicates associated with that table.
        """
        filters = []
        NodeOps.map_leaves(self, lambda l: filters.append(l.info.get("filter")))
        filters = list(filter(lambda s: s is not None, filters))
        return filters

    def GetEqualityFilters(self):
        """Returns the list of equality filter predicates.

        These expressions are of the form rel.attr = VALUE.
        """
        eq_filters = self.info.get("eq_filters")
        if eq_filters is None:
            filters = self.GetFilters()
            pattern = re.compile("[a-z][\da-z_]*\.[\da-z_]*\ = [\da-z_]+")
            equality_conds = []
            for clause in filters:
                equality_conds.extend(pattern.findall(clause))
            eq_filters = list(set(equality_conds))
            self.info["eq_filters"] = eq_filters
        return eq_filters

    def GetFilteredAttributes(self):
        """Returns the list of filtered attributes ([<alias>.<attr>])."""
        attrs = self.info.get("filtered_attributes")
        if attrs is None:
            attrs = []
            filters = self.GetFilters()
            # Look for <alias>.<attr>.
            pattern = re.compile("[a-z][\da-z_]*\.[\da-z_]+")
            for clause in filters:
                attrs.extend(pattern.findall(clause))
            attrs = list(set(attrs))
            self.info["filtered_attributes"] = attrs
        return attrs

    def KeepRelevantJoins(self, all_join_conds):
        """Returns join conditions relevant to this Node."""
        aliases = self.leaf_ids(alias_only=True)

        def _KeepRelevantJoins(s):
            splits = s.split("=")
            l, r = splits[0].strip(), splits[1].strip()
            l_alias = l.split(".")[0]
            r_alias = r.split(".")[0]
            return l_alias in aliases and r_alias in aliases

        joins = list(filter(_KeepRelevantJoins, all_join_conds))
        return joins

    @functools.lru_cache(maxsize=8)
    def leaf_ids(self, with_alias=True, return_depths=False, alias_only=False):
        """
        SQL: SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id;

        node.leaf_ids()
        -> ["orders AS o", "customers AS c"]

        node.leaf_ids(alias_only=True)
        -> ["o", "c"]

        node.leaf_ids(with_alias=False)
        -> ["orders", "customers"]

        node.leaf_ids(return_depths=True)
        -> (["orders AS o", "customers AS c"], [1, 1])
        """
        ids = []
        if not return_depths:
            NodeOps.map_leaves(
                self, lambda l: ids.append(l.get_table_id(with_alias, alias_only))
            )
            return ids

        depths = []

        def _Helper(leaf, depth):
            ids.append(leaf.get_table_id(with_alias, alias_only))
            depths.append(depth)

        NodeOps.map_leaves_with_depth(self, _Helper)
        return ids, depths

    def Copy(self):
        """Returns a deep copy of self."""
        return copy.deepcopy(self)

    def CopyLeaves(self):
        """Returns a list of deep copies of the leaf nodes."""
        leaves = []
        NodeOps.map_leaves(self, lambda leaf: leaves.append(copy.deepcopy(leaf)))
        return leaves

    def GetLeaves(self):
        """Returns a list of references to the leaf nodes."""
        leaves = []
        NodeOps.map_leaves(self, lambda leaf: leaves.append(leaf))
        return leaves

    def IsJoin(self):
        return "Join" in self.node_type or self.node_type == "Nested Loop"

    def IsScan(self):
        return "Scan" in self.node_type

    def HasEqualityFilters(self):
        return len(self.GetEqualityFilters()) > 0

    def ToScanOp(self, scan_op):
        """Retrieves a deep copy of self with scan_op assigned."""
        assert not self.children, "This node must be a leaf."
        copied = self._leaf_scan_op_copies.get(scan_op)
        if copied is None:
            copied = copy.deepcopy(self)
            copied.node_type = scan_op
            self._leaf_scan_op_copies[scan_op] = copied
        return copied

    def to_sql(self, all_join_conds, with_filters=True, with_select_exprs=False):
        # Join and filter predicates.
        joins = self.KeepRelevantJoins(all_join_conds)
        if with_filters:
            filters = self.GetFilters()
        else:
            filters = []
        # FROM.
        from_str = self.leaf_ids()
        from_str = ", ".join(from_str)
        # SELECT.
        if with_select_exprs:
            select_exprs = self.GetSelectExprs()
        else:
            select_exprs = []
        select_str = "*" if len(select_exprs) == 0 else ",".join(select_exprs)

        if len(filters) > 0 and len(joins) > 0:
            sql = "SELECT {} FROM {} WHERE {} AND {};".format(
                select_str, from_str, " AND ".join(joins), " AND ".join(filters)
            )
        elif len(joins) > 0:
            sql = "SELECT {} FROM {} WHERE {};".format(
                select_str, from_str, " AND ".join(joins)
            )
        elif len(filters) > 0:
            sql = "SELECT {} FROM {} WHERE {};".format(
                select_str, from_str, " AND ".join(filters)
            )
        else:
            sql = "SELECT {} FROM {};".format(select_str, from_str)
        return sql

    @functools.lru_cache(maxsize=2)
    def hint_str(self, with_physical_hints=False):
        """Produces a plan hint such that query_plan (Node) is respected."""
        scans = []
        joins = []

        def helper(t):
            node_type = t.node_type.replace(" ", "")
            # PG uses the former & the extension expects the latter.
            node_type = node_type.replace("NestedLoop", "NestLoop")
            node_type = node_type.replace("BitmapHeapScan", "BitmapScan")
            node_type = node_type.replace("BitmapIndexScan", "BitmapScan")
            if t.IsScan():
                scans.append(node_type + "(" + t.table_alias + ")")
                return [t.table_alias], t.table_alias
            rels = []  # Flattened
            leading = []  # Hierarchical
            for child in t.children:
                a, b = helper(child)
                rels.extend(a)
                leading.append(b)
            joins.append(node_type + "(" + " ".join(rels) + ")")
            return rels, leading

        _, leading_hierarchy = helper(self)

        leading = (
            "Leading("
            + str(leading_hierarchy)
            .replace("'", "")
            .replace("[", "(")
            .replace("]", ")")
            .replace(",", "")
            + ")"
        )
        if with_physical_hints:
            # Reverse the join hints to print largest/outermost joins first.
            atoms = joins[::-1] + scans + [leading]
        else:
            atoms = [leading]  # Join order hint only.
        query_hint = "\n ".join(atoms)
        return "/*+ " + query_hint + " */"

    def filter_scans_joins(self):
        """
        Filters the trees: keeps only the scan and join nodes.
        Input nodes are copied and are not modified in-place.
        Examples of removed nodes (all unary): Aggregate, Gather, Hash, Materialize.
        """

        def _filter(_node):
            if not _node.IsScan() and not _node.IsJoin():
                assert len(_node.children) == 1, _node
                return _filter(_node.children[0])
            _node.children = [_filter(c) for c in _node.children]
            return _node

        new_node = _filter(self.Copy())
        # Save top-level node's info and cost (which might be latency value
        # from actual execution), since the top-level node may get filtered away.
        new_node.info = copy.deepcopy(self.info)
        new_node.cost = self.cost
        new_node.actual_time_ms = self.actual_time_ms
        return new_node

    def gather_filter_info(self, with_alias=True, alias_only=False):
        """For each node, gather leaf filters into root as node.info['all_filters']."""
        d = {}

        def f(leaf):
            if "filter" in leaf.info:
                table_id = leaf.get_table_id(with_alias, alias_only)
                assert table_id not in d, (leaf.info, table_id, d)
                d[table_id] = leaf.info["filter"]

        NodeOps.map_leaves(self, f)
        self.info["all_filters"] = d

    def is_filter_scan(self):
        """
        Checks if a node represents a scan with equality filters.
        """
        return self.HasEqualityFilters() and self.IsScan()

    def is_small_scan(self):
        """
        Determines if a scan node is considered "small" based on table name.
        Currently uses a heuristic treating tables ending in '_type' as small.
        A more robust approach would involve analyzing Postgres source code for row count or cost thresholds.
        """
        return self.table_name.endswith("_type")

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return str(self) < str(other)


class NodeOps:
    """Utility class for common operations on Node trees."""

    @staticmethod
    def map_leaves(node, func):
        """Applies func: node -> U over each leaf of 'node'."""

        def f(n):
            if len(n.children) == 0:
                func(n)

        NodeOps.map_node(node, f)

    @staticmethod
    def map_leaves_with_depth(node, func, depth=0):
        """Applies func: (node, depth) -> U over each leaf of 'node'.
        The current node has a depth of 'depth' (defaults to 0). Any node in
        node.children has a depth of depth+1, etc.
        """
        assert node is not None

        def f(n, d):
            if len(n.children) == 0:
                func(n, d)

        NodeOps.map_node_with_depth(node, f, depth)

    @staticmethod
    def map_node(node, func):
        """Applies func over each subnode of 'node' (preorder traversal)."""
        func(node)
        for c in node.children:
            NodeOps.map_node(c, func)

    @staticmethod
    def map_node_with_depth(node, func, depth=0):
        """Applies func: (node, depth) -> U over each subnode of 'node'.

        The current node has a depth of 'depth' (defaults to 0). Any node in
        node.children has a depth of depth+1, etc.
        """
        func(node, depth)
        for c in node.children:
            NodeOps.map_node_with_depth(c, func, depth + 1)

    @staticmethod
    def exists_join_edge_in_graph(node1: Node, node2: Node, join_graph):
        """Checks if two nodes are connected via an edge in the join graph."""
        assert isinstance(join_graph, nx.Graph), join_graph
        leaves1 = node1.leaf_ids(alias_only=True)
        leaves2 = node2.leaf_ids(alias_only=True)
        edges = join_graph.edges
        for name1 in leaves1:
            for name2 in leaves2:
                if (name1, name2) in edges:
                    return True
        return False

    @staticmethod
    def rewrite_as_generic_joinscan(nodes: List[Node]):
        """Rewrite all scan/join nodes into generic 'Scan'/'Join' labels."""

        def f(node: Node):
            op = node.node_type
            if "Scan" in op:
                node.node_type = "Scan"
            elif "Join" in op or op == "Nested Loop":
                node.node_type = "Join"

        for node in nodes:
            NodeOps.map_node(node, f)

    @staticmethod
    def get_all_subtrees(nodes: List[Node]):
        """Returns all subtrees (Node roots) including leaves."""
        trees = []

        def _fn(node: Node, trees):
            trees.append(node)
            for c in node.children:
                # Skip Bitmap Index Scans under Bitmap Heap Scan
                if (node.node_type != "Bitmap Heap Scan") or (
                    c.node_type != "Bitmap Index Scan"
                ):
                    _fn(c, trees)

        if isinstance(nodes, Node):
            nodes = [nodes]
        for node in nodes:
            _fn(node, trees)
        return trees

    @staticmethod
    def get_all_subtrees_no_leaves(nodes: List[Node]):
        """Returns all subtrees excluding leaves (treat Bitmap Heap Scan as leaf)."""
        trees = []

        def _fn(node: Node, trees):
            if len(node.children) and (node.node_type != "Bitmap Heap Scan"):
                trees.append(node)
                for c in node.children:
                    _fn(c, trees)

        if isinstance(nodes, Node):
            nodes = [nodes]
        for node in nodes:
            _fn(node, trees)
        return trees

    @staticmethod
    def filter_exprs_by_alias(exprs, table_alias):
        """Filters expressions by a given table alias."""
        pattern = re.compile(r".*\(?\b{}\b\..*\)?".format(table_alias))
        return list(filter(pattern.match, exprs))

    @staticmethod
    def is_join_combination_ok(
        join_type, left, right, avoid_eq_filters=False, use_plan_restrictions=True
    ):
        """
        Checks if a hinted join is likely to be accepted by Postgres.
        Postgres may silently reject or rewrite hints due to internal implementation constraints.
        This function uses empirically determined checks, which may be more conservative than Postgres' actual criteria.

        Args:
            join_type (str): The type of join operation planned (e.g., 'Nested Loop', 'Hash Join').
            left (Node): Left child node of the join with its scan type assigned.
            right (Node): Right child node of the join with its scan type assigned.
            avoid_eq_filters (bool): If True, avoids certain equality filter scans for Ext-JOB planning.
            use_plan_restrictions (bool): If True, applies empirical join restrictions.

        Returns:
            bool: True if the join combination is likely to be respected by Postgres, False otherwise.

        Original function name: IsJoinCombinationOk
        """
        if not use_plan_restrictions:
            return True
        if join_type == "Nested Loop":
            return NodeOps.is_nest_loop_legal(left, right)
        # Avoid problematic info_type + equality filter combinations for Ext-JOB hints
        if avoid_eq_filters:
            if right.table_name == "info_type" and right.is_filter_scan():
                return False
            if left.table_name == "info_type" and left.is_filter_scan():
                return False
        if join_type == "Hash Join":
            return NodeOps.is_hash_join_ok(left, right)
        return True

    @staticmethod
    def is_nest_loop_legal(left, right):
        """
        decide whether a hinted Nested Loop join is "legal" (accepted by PG) given the scan/join types of its children

        Args:
            left (Node): Left child node of the join.
            right (Node): Right child node of the join.

        Returns:
            bool: True if the Nested Loop configuration is valid, False otherwise.

        Original function name: IsNestLoopOk
        """
        l_op = left.node_type
        r_op = right.node_type
        if (l_op, r_op) not in BaseConfig.NestedLoop_WHITE_LIST:
            return False
        # Ensure Seq Scan side is small for specific combinations
        if (l_op, r_op) == ("Seq Scan", "Nested Loop"):
            return left.is_small_scan()
        if (l_op, r_op) in [("Index Scan", "Seq Scan"), ("Nested Loop", "Seq Scan")]:
            return right.is_small_scan()
        return True

    @staticmethod
    def is_hash_join_ok(left, right):
        """
        Checks if a Hash Join hint is likely to be respected by Postgres based on empirical rules.

        Args:
            left (Node): Left child node of the join.
            right (Node): Right child node of the join.

        Returns:
            bool: True if the Hash Join configuration is valid, False otherwise.

        Original function name: IsHashJoinOk
        """
        l_op = left.node_type
        r_op = right.node_type
        if (l_op, r_op) == ("Index Scan", "Hash Join"):
            l_exprs = left.GetSelectExprs()
            r_exprs = right.GetSelectExprs()
            return left.is_small_scan() and (len(l_exprs) or len(r_exprs))
        return True

    @staticmethod
    def enumerate_scan_ops(node, scan_ops):
        """
        Yields possible scan operations for a given node, excluding 'Index Only Scan'.

        Args:
            node: Node object to process.
            scan_ops (list): List of possible scan operations.

        Yields:
            Node: Node with applied scan operation or original node if not a scan.

        Original function name: EnumerateScanOps
        """
        if not node.IsScan():
            yield node
        else:
            for scan_op in scan_ops:
                if scan_op == "Index Only Scan":
                    continue
                yield node.ToScanOp(scan_op)

    @staticmethod
    def enumerate_join_with_ops(
        left,
        right,
        join_ops,
        scan_ops,
        avoid_eq_filters=False,
        use_plan_restrictions=True,
    ):
        """
        Yields all valid join operations combining scan operations on left and right nodes.

        Args:
            left (Node): Left child node of the join.
            right (Node): Right child node of the join.
            join_ops (list): List of possible join operations.
            scan_ops (list): List of possible scan operations.
            avoid_eq_filters (bool): If True, avoids equality filter scans.
            use_plan_restrictions (bool): If True, applies join restrictions.

        Yields:
            Node: Valid join node with assigned children.
        """
        for join_op in join_ops:
            for l in NodeOps.enumerate_scan_ops(left, scan_ops):
                for r in NodeOps.enumerate_scan_ops(right, scan_ops):
                    if not NodeOps.is_join_combination_ok(
                        join_op, l, r, avoid_eq_filters, use_plan_restrictions
                    ):
                        continue
                    join = Node(join_op)
                    join.children = [l, r]
                    yield join


class WorkloadInfo:
    """Stores sets of possible relations/aliases/join types, etc.

    From a list of all Nodes, parse
    - all relation names
    - all join types
    - all scan types.
    These can also be specified manually for a workload.

    Attributes:
      rel_names, rel_ids, scan_types, join_types, all_ops: ndarray of sorted
        strings.
    """

    def __init__(self, nodes: List[Node]):

        # assgined later
        self.table_num_rows = None

        self.rel_names_set = set()
        self.rel_ids_set = set()
        self.scan_types_set = set()
        self.join_types_set = set()
        self.all_ops_set = set()
        self.all_attributes_set = set()
        self.all_filters = collections.defaultdict(set)

        for node in nodes:
            self._fill(node, node)

        self.nodes = nodes
        self.rel_names = np.asarray(sorted(list(self.rel_names_set)))
        self.rel_ids = np.asarray(sorted(list(self.rel_ids_set)))
        self.scan_types = np.asarray(sorted(list(self.scan_types_set)))
        self.join_types = np.asarray(sorted(list(self.join_types_set)))
        self.all_ops = np.asarray(sorted(list(self.all_ops_set)))
        self.all_attributes = np.asarray(sorted(list(self.all_attributes_set)))
        self.join_edge_set = set()

        # Clean up temporary sets
        del self.rel_names_set
        del self.rel_ids_set
        del self.scan_types_set
        del self.join_types_set
        del self.all_ops_set
        del self.all_attributes_set

    def _fill(self, root: Node, node: Node):
        self.all_ops_set.add(node.node_type)

        if node.table_name is not None:
            self.rel_names_set.add(node.table_name)
            self.rel_ids_set.add(node.get_table_id())

        if node.info and "filter" in node.info:
            table_id = node.get_table_id()
            self.all_filters[table_id].add(node.info["filter"])

        if node.info and "sql_str" in node.info:
            # We want "all" attributes but as an optimization, we keep the
            # attributes that are known to be filter-able.
            attrs = node.GetFilteredAttributes()
            self.all_attributes_set.update(attrs)

        if "Scan" in node.node_type:
            self.scan_types_set.add(node.node_type)
        elif node.IsJoin():
            self.join_types_set.add(node.node_type)

        for c in node.children:
            self._fill(root, c)

    def SetPhysicalOps(self, join_ops: List, scan_ops: List):
        old_scans = self.scan_types
        old_joins = self.join_types
        if scan_ops is not None:
            self.scan_types = np.asarray(sorted(list(scan_ops)))
        if join_ops is not None:
            self.join_types = np.asarray(sorted(list(join_ops)))
        new_all_ops = [
            op for op in self.all_ops if op not in old_scans and op not in old_joins
        ]
        new_all_ops = new_all_ops + list(self.scan_types) + list(self.join_types)
        if len(self.all_ops) != len(new_all_ops):
            print(
                "Search space (old=parsed from query nodes; new=manually set in search space):"
            )
            print("old:", old_scans, old_joins, self.all_ops)
            print("new:", self.scan_types, self.join_types, self.all_ops)
        self.all_ops = np.asarray(sorted(list(set(new_all_ops))))

    def WithJoinGraph(self, join_graph: Dict):
        """Transforms { table -> neighbors } into internal representation."""
        self.join_edge_set = set()
        for t1, neighbors in join_graph.items():
            for t2 in neighbors:
                self.join_edge_set.add((t1, t2))
                self.join_edge_set.add((t2, t1))

    def Copy(self):
        return copy.deepcopy(self)

    def HasPhysicalOps(self):
        if not np.array_equal(self.scan_types, ["Scan"]):
            return True
        if not np.array_equal(self.join_types, ["Join"]):
            return True
        return False

    def __repr__(self):
        fmt = (
            "rel_names: {}\nrel_ids: {}\nscan_types: {}\n"
            "join_types: {}\nall_ops: {}\nall_attributes: {}"
        )
        return fmt.format(
            self.rel_names,
            self.rel_ids,
            self.scan_types,
            self.join_types,
            self.all_ops,
            self.all_attributes,
        )


def MakeTestQuery():
    a_join_b = Node("Join")
    a_join_b.children = [
        Node("Scan", table_name="A").with_alias("a"),
        Node("Scan", table_name="B").with_alias("b"),
    ]
    query_node = Node("Join")
    query_node.children = [a_join_b, Node("Scan", table_name="C").with_alias("c")]
    query_node.info["query_name"] = "test"
    query_node.info["sql_str"] = (
        "SELECT * FROM a, b, c WHERE a.id = b.id AND a.id = c.id;"
    )

    # Simple check: Copy() deep-copies the children nodes.
    copied = query_node.Copy()
    assert id(copied) != id(query_node)
    for i in range(2):
        assert id(copied.children[i]) != id(query_node.children[i])

    return query_node
