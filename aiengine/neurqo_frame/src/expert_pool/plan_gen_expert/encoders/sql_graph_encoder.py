import numpy as np
import torch
from common import workload
import networkx as nx


class SimQueryFeaturizer:
    """Implements the query featurizer.

        Query node -> [ multi-hot of what tables are present ]
                    * [ each-table's selectivities ]
    """

    def __init__(self, workload_info: workload.WorkloadInfo):
        self.workload_info = workload_info

    def __call__(self, node: workload.Node):
        vec = np.zeros(self.dims, dtype=np.float32)

        # Joined tables: [table: 1].
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = 1.0

        # Filtered tables.
        table_id_to_name = lambda table_id: table_id.split(' ')[0]  # Hack.

        for rel_id, est_rows in node.info['all_filters_est_rows'].items():
            if rel_id not in joined:
                # Due to the way we copy Nodes and populate this info field,
                # leaf_ids() might be a subset of info['all_filters_est_rows'].
                continue

            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            total_rows = self.workload_info.table_num_rows[table_id_to_name(rel_id)]

            # # NOTE: without ANALYZE, for some reason this predicate is
            # # estimated to have 703 rows, whereas the table only has 4 rows:
            # #   (kind IS NOT NULL) AND ((kind)::text <> 'production
            # #   companies'::text)
            # # With ANALYZE run, this assert passes.
            # assert est_rows >= 0 and est_rows <= total_rows, (node.info,
            #                                                   est_rows,
            #                                                   total_rows)
            # Assertion is triggered with STACK query, instead replaced by clamping
            est_rows = max(0, est_rows)
            est_rows = min(est_rows, total_rows)

            vec[idx] = est_rows / total_rows
        return vec

    def PerturbQueryFeatures(self, query_feat, distribution):
        """Randomly perturbs a query feature vec returned by __call__()."""
        selectivities = query_feat
        # Table: for each chance of each joined table being perturbed:
        #     % of original query features kept
        #     mean # tables scaled
        #
        #   0.5: ~3% original; mean # tables scaled 3.6
        #   0.3: ~10.5% original; mean # tables scaled 2.1
        #   0.25: ~13.9-16.6% original; mean # tables scaled 1.8-1.9
        #   0.2: ~23.6% original; mean # tables scaled 1.5
        #
        # % kept original:
        #   ((multipliers > 1).sum(1) == 0).sum().float() / len(multipliers)
        # Mean # tables scaled:
        #   (multipliers > 1).sum(1).float().mean()
        #
        # "Default": chance = 0.25, unif = [0.5, 2].
        chance, unif = distribution

        should_scale = torch.rand(selectivities.shape,
                                  device=selectivities.device) < chance
        # The non-zero entries are joined tables.
        should_scale *= (selectivities > 0)
        # Sample multipliers ~ Unif[l, r].
        multipliers = torch.rand(
            selectivities.shape,
            device=selectivities.device) * (unif[1] - unif[0]) + unif[0]
        multipliers *= should_scale
        # Now, the 0 entries mean "should not scale", which needs to be
        # translated into using a multiplier of 1.
        multipliers[multipliers == 0] = 1
        # Perturb.
        new_selectivities = torch.clamp(selectivities * multipliers, max=1)
        return new_selectivities

    @property
    def dims(self):
        return len(self.workload_info.rel_ids)


class QueryFeaturizer:
    """Concat [join graph] [table T: log(est_rows(T) + 1)/log(max_rows)]."""

    def __init__(self, workload_info):
        self.workload_info = workload_info
        self.MAX_ROW_COUNT = max(workload_info.table_num_rows.values())
        self.table_id_to_name = lambda table_id: table_id.split(' ')[0]
        self.table_id_to_alias = lambda table_id: table_id.split(' ')[-1]
        print(
            'QueryFeaturizer, {} rel_ids'.format(len(
                self.workload_info.rel_ids)), self.workload_info.rel_ids)

    def __call__(self, node):
        vec = np.zeros_like(self.workload_info.rel_ids, dtype=np.float32)

        # Joined tables.
        # [table: row count of each table]
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = self.workload_info.table_num_rows[self.table_id_to_name(
                rel_id)]

        # Filtered tables.
        for rel_id, est_rows in node.info['all_filters_est_rows'].items():
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            assert vec[idx] > 0, (node, node.info['all_filters_est_rows'])
            total_rows = self.workload_info.table_num_rows[
                self.table_id_to_name(rel_id)]

            # NOTE: without ANALYZE, for some reason this predicate is
            # estimated to have 703 rows, whereas the table only has 4 rows:
            #   (kind IS NOT NULL) AND ((kind)::text <> 'production
            #   companies'::text)
            # With ANALYZE run, this assert passes.
            assert est_rows >= 0 and est_rows <= total_rows, (node.info,
                                                              est_rows,
                                                              total_rows)
            sel = est_rows / total_rows
            # TODO: an additional signal is to provide selectivity.
            vec[idx] = est_rows

        vec = np.log(vec + 1.0) / np.log(self.MAX_ROW_COUNT)

        # Query join graph features: adjacency matrix.
        query_join_graph = node.GetOrParseJoinGraph()
        all_aliases = [
            self.table_id_to_alias(table)
            for table in self.workload_info.rel_ids
        ]
        adj_matrix = nx.to_numpy_array(query_join_graph, nodelist=all_aliases)
        # Sufficient to grab the upper-triangular portion (since graph is
        # undirected).  k=1 means don't grab the diagnoal (all 1s).
        triu = adj_matrix[np.triu_indices(len(all_aliases),
                                          k=1)].astype(np.float32)

        features = np.concatenate((triu, vec), axis=None)
        return features
