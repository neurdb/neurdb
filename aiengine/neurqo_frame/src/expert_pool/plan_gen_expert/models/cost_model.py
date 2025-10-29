from common import hyperparams
from parser import plan_parser, plan_node
from db.pg_conn import PostgresConnector


def ContainsPhysicalHints(hint_str):
    HINTS = [
        'SeqScan',
        'IndexScan',
        'IndexOnlyScan',
        'NestLoop',
        'HashJoin',
        'MergeJoin',
    ]
    for hint in HINTS:
        if hint in hint_str:
            return True
    return False


def get_pg_estimated_cost(cursor, sql, hint, check_hint_used=False):
    # GEQO must be disabled for hinting larger joins to work.
    cursor.set_geqo_exp('off')
    node0, _ = plan_node.PGToNodeHelper.sql_to_plan_node(cursor=cursor, sql=sql, comment=hint)
    # This copies top-level node's cost (e.g., Aggregate) to the new top level node (a Join).
    node = node0.filter_scans_joins()

    cursor.set_geqo_exp('default')
    if check_hint_used:
        expected = hint
        actual = node.hint_str(with_physical_hints=ContainsPhysicalHints(hint))
        assert expected == actual, 'Expected={}\nActual={}, actual node:\n{}\nSQL=\n{}'.format(expected, actual, node,
                                                                                               sql)

    return node.cost


class CardEst:
    """Base class for cardinality estimators."""

    def __call__(self, node, join_conds):
        raise NotImplementedError()


class PostgresCardEst(CardEst):

    def __init__(self, cursor: PostgresConnector):
        self._cache = {}
        self.cursor = cursor

    def _HashKey(self, node):
        """Computes a hash key based on the logical contents of 'node'.

        Specifically, hash on the sorted sets of table IDs and their filters.

        NOTE: Postgres can produce slightly different cardinality estimates
        when all being equal but just the FROM list ordering tables
        differently.  Here, we ignore this slight difference.
        """
        sorted_filters = '\n'.join(sorted(node.GetFilters()))
        sorted_leaves = '\n'.join(sorted(node.leaf_ids()))
        return sorted_leaves + sorted_filters

    def __call__(self, node, join_conds):
        key = self._HashKey(node)
        card = self._cache.get(key)
        if card is None:
            sql_str = node.to_sql(join_conds)
            _, json_dict = plan_node.sql_to_plan_node(self.cursor, sql_str)
            card = json_dict['Plan']['Plan Rows']
            self._cache[key] = card
        return card


class CostModelBase:
    """Base class for query plan cost models."""

    @classmethod
    def Params(cls):
        """
        Defines configuration parameters for the CostModelBase.

        Returns:
            hyperparams.InstantiableParams: Configured parameters for the class.

        Original method name: Params
        """
        p = hyperparams.InstantiableParams(cls)
        p.define('cost_physical_ops', False, 'Whether to cost physical operators or only join orders.')
        return p

    def __init__(self, params):
        """
        Initializes the CostModelBase with given parameters.

        Args:
            params: Configuration parameters for the cost model.
        """
        self.params = params.copy()

    def __call__(self, node, join_conds):
        """
        Computes the cost of a query plan node with associated join conditions.
        Filter information in leaf nodes should be considered.

        Args:
            node: balsa.Node representing the query plan or subplan.
            join_conds: Join conditions associated with the node.

        Raises:
            NotImplementedError: This is an abstract method.

        Original method name: __call__
        """
        raise NotImplementedError('Abstract method')

    def score_with_sql(self, node, sql):
        """
        Scores a query plan node using its hint string and SQL query.

        Args:
            node: balsa.Node representing the query plan.
            sql (str): SQL query string to evaluate.

        Raises:
            NotImplementedError: This is an abstract method.

        Original method name: ScoreWithSql
        """
        raise NotImplementedError('Abstract method')


class NullCost(CostModelBase):
    """A cost model that assigns zero cost to any query plan."""

    def __call__(self, node, join_conds):
        """
        Assigns a cost of 0 to the given query plan node.

        Args:
            node: balsa.Node representing the query plan or subplan.
            join_conds: Join conditions (ignored in this model).

        Returns:
            int: Always returns 0.

        Original method name: __call__
        """
        return 0

    def score_with_sql(self, node, sql):
        """
        Assigns a cost of 0 to the query plan node and SQL query.

        Args:
            node: balsa.Node representing the query plan.
            sql (str): SQL query string (ignored in this model).

        Returns:
            int: Always returns 0.

        Original method name: ScoreWithSql
        """
        return 0


class PostgresCost(CostModelBase):
    """Cost model based on Postgres' internal cost estimates."""

    def __init__(self, params, cursor: PostgresConnector):
        super().__init__(params)
        self.cursor = cursor

    def __call__(self, node, join_conds):
        """
        Computes the cost of a query plan node using Postgres' cost estimation.
        Note: Postgres may reject a HashJoin hint with "SELECT * ..." but accept it
        with "SELECT min(...) ...". This can affect cost estimates during data collection
        for learning-based models, though the impact is typically minimal.

        Args:
            node: balsa.Node representing the query plan or subplan.
            join_conds: Join conditions associated with the node.

        Returns:
            float: Estimated cost from Postgres for the query plan.

        Original method name: __call__
        """
        sql_str = node.to_sql(join_conds, with_select_exprs=True)
        return self.score_with_sql(node, sql_str)

    def score_with_sql(self, node, sql):
        """
        Scores a query plan node using Postgres' cost estimation for the given SQL and hints.

        Args:
            node: balsa.Node representing the query plan.
            sql (str): SQL query string to evaluate.

        Returns:
            float: Estimated cost from Postgres.

        Original method name: ScoreWithSql
        """
        p = self.params
        cost = get_pg_estimated_cost(
            cursor=self.cursor,
            sql=sql,
            hint=node.hint_str(with_physical_hints=p.cost_physical_ops),
            # check_hint_used=True,
            check_hint_used=False,
        )
        return cost


class MinCardCost(CostModelBase):
    """
    A cost model that minimizes intermediate result cardinality (C_out).
    Suitable for local join order planning, ignoring physical scan/join methods.

    Cost calculation:
        - Base table: C(T) = |T|
        - Filtered base table: C(T) = |filter(T)|
        - Join: C(T) = C(T1) + C(T2) + |T|

    References:
        - https://arxiv.org/pdf/2005.03328.pdf
        - Neumann et al., "Query Optimization in the Big Data Era" (https://dl.acm.org/doi/pdf/10.1145/1559845.1559889)
    """

    def __init__(self, params, cursor: PostgresConnector):
        """
        Initializes the MinCardCost model with a cardinality estimator.

        Args:
            params: Configuration parameters for the cost model.
        """
        super().__init__(params)
        self.card_est = PostgresCardEst(cursor)

    def __call__(self, node, join_conds):
        """
        Computes the cost of a query plan node based on cardinality.

        Args:
            node: balsa.Node representing the query plan or subplan.
            join_conds: Join conditions associated with the node.

        Returns:
            float: Estimated cost based on cardinality.

        Original method name: __call__
        """
        return self.score(node, join_conds)

    def get_model_cardinality(self, node, join_conds):
        """
        Estimates the cardinality of a query plan node.

        Args:
            node: balsa.Node representing the query plan or subplan.
            join_conds: Join conditions associated with the node.

        Returns:
            int: Estimated number of tuples for the node.

        Original method name: GetModelCardinality
        """
        joins = node.KeepRelevantJoins(join_conds)
        if len(joins) == 0 and len(node.GetFilters()) == 0:
            return self.get_base_rel_cardinality(node)
        return self.card_est(node, joins)

    def get_base_rel_cardinality(self, node):
        """
        Retrieves the cardinality of a base table.

        Args:
            node: balsa.Node representing a base table.

        Returns:
            int: Number of rows in the base table.

        Raises:
            AssertionError: If node.table_name is None.

        Original method name: GetBaseRelCardinality
        """
        assert node.table_name is not None, node
        return GetAllTableNumRows([node.table_name])[node.table_name]

    def score(self, node, join_conds):
        """
        Computes the cost of a query plan node based on cardinality, recursively for joins.

        Args:
            node: balsa.Node representing the query plan or subplan.
            join_conds: Join conditions associated with the node.

        Returns:
            float: Total cost based on cardinality.

        Original method name: Score
        """
        if node._card:
            return node._card

        card = self.get_model_cardinality(node, join_conds)
        if node.IsScan():
            node._card = card
        else:
            assert node.IsJoin(), node
            c_t1 = self.score(node.children[0], join_conds)
            c_t2 = self.score(node.children[1], join_conds)
            node._card = card + c_t1 + c_t2

        return node._card

# def GetCardinalityEstimateFromPg(sql):
#     _, json_dict = plan_node.SqlToPlanNode(sql)
#     return json_dict['Plan']['Plan Rows']
