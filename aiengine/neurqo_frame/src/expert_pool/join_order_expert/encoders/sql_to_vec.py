from db.pg_conn import PostgresConnector

# Import from local package
from expert_pool.join_order_expert.encoders.job_parser import TargetTable, FromTable, Comparison
import numpy as np


class Sql2Vec:
    """
    Encapsulates SQL parsing + selectivity/vectorization logic.
    - No global state: per-instance caches (e.g., column_id) live here.
    - External dependencies (pg_runner, config) are provided via __init__.
    """
    def __init__(self, db_cli: PostgresConnector, config=None):
        self.db_cli = db_cli
        self.config = config
        # formerly-global column_id map moved here
        self._column_id = {}

    # ---- local helpers (moved from globals) ----
    def _get_column_id(self, column: str) -> int:
        if column not in self._column_id:
            self._column_id[column] = len(self._column_id)
        return self._column_id[column]

    # ---- main API ----
    def to_vec(self, sql: str):
        """
        Example output vec = [join_matrix(flattened), column/alias selectivity]
             e.g., [0,1,0,1,0,0,0,0,0, 0.2,0.3,0.0]
             (join edges + filter selectivity values, mostly 0 except for used columns)
        Returns:
          - np.ndarray feature vector (join matrix + selectivity/counts)
          - List[str] alias roots (sorted by aliasname2id)
          - List[Tuple[str, str]] joins_with_predicate (sorted)
          - List[Tuple[str, str]] joins (sorted)
        """
        if self.db_cli is None:
            raise RuntimeError("pg_runner is required for selectivity evaluation.")

        # lazy import to keep module import-time light
        from psqlparse import parse_dict

        parse_result = parse_dict(sql)[0]["SelectStmt"]
        target_table_list = [TargetTable(x["ResTarget"]) for x in parse_result["targetList"]]
        from_table_list = [FromTable(x["RangeVar"]) for x in parse_result["fromClause"]]

        if len(from_table_list) < 2:
            # keep original behavior
            print(f"table list = {from_table_list}, and is <2, return none ")
            return None

        aliasname2fullname = {}
        id2aliasname = self.config.id2aliasname
        aliasname2id = self.config.aliasname2id

        join_set = set()
        aliasnames_root_set = {x.getAliasName() for x in from_table_list}

        alias_selectivity = np.asarray([0.0] * len(id2aliasname), dtype=float)
        aliasname2fromtable = {}
        for table in from_table_list:
            aliasname2fromtable[table.getAliasName()] = table
            aliasname2fullname[table.getAliasName()] = table.getFullName()

        comparison_list = [Comparison(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]

        join_matrix = np.zeros((len(id2aliasname), len(id2aliasname)), dtype=float)
        count_selectivity = np.asarray([0.0] * self.config.max_column, dtype=float)
        has_predicate = set()
        join_with_pred_set = set()

        # collect join edges + selectivities
        for comparison in comparison_list:
            if len(comparison.aliasname_list) == 2:
                left_aliasname, right_aliasname = comparison.aliasname_list
                idx0 = aliasname2id[left_aliasname]
                idx1 = aliasname2id[right_aliasname]
                edge = (left_aliasname, right_aliasname) if idx0 < idx1 else (right_aliasname, left_aliasname)
                join_set.add(edge)
                join_matrix[idx0][idx1] = 1.0
                join_matrix[idx1][idx0] = 1.0
            else:
                left_aliasname = comparison.aliasname_list[0]
                sel = self.db_cli.get_selectivity(str(aliasname2fromtable[left_aliasname]), str(comparison))
                alias_selectivity[aliasname2id[left_aliasname]] += sel
                has_predicate.add(left_aliasname)

                col_id = self._get_column_id(comparison.column)
                count_selectivity[col_id] += sel

        for ajoin in join_set:
            if ajoin[0] in has_predicate or ajoin[1] in has_predicate:
                join_with_pred_set.add(ajoin)

        # build the feature vector
        if self.config.max_column == 40:
            vec = np.concatenate((join_matrix.flatten(), alias_selectivity))
        else:
            vec = np.concatenate((join_matrix.flatten(), count_selectivity))

        # ---- make everything collatable & deterministic ----
        alias_list = sorted(aliasnames_root_set, key=lambda a: aliasname2id[a])
        join_list = sorted(join_set, key=lambda e: (aliasname2id[e[0]], aliasname2id[e[1]]))
        join_list_with_predicate = sorted(join_with_pred_set, key=lambda e: (aliasname2id[e[0]], aliasname2id[e[1]]))

        return vec, alias_list, join_list_with_predicate, join_list
