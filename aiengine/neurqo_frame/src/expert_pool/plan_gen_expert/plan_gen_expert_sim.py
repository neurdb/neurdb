import collections
import copy
import hashlib
import os
import pickle
import time
from parser import plan_node
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from absl import app, logging
from common import hyperparams, workload
from common.base_config import BaseConfig
from db.pg_conn import PostgresConnector
from expert_pool.plan_gen_expert.dataset import expert_datasets

# Import from local package
from expert_pool.plan_gen_expert.encoders import (
    plan_graph_encoder,
)
from expert_pool.plan_gen_expert.encoders import (
    sql_graph_encoder as query_graph_encoder,
)

# Import from local package for treeconv
from expert_pool.plan_gen_expert.models import cost_model, treeconv
from expert_pool.plan_gen_expert.search_alg.beam import BMOptimizer
from expert_pool.plan_gen_expert.search_alg.dymanic_progm import DynamicProgramming
from expert_pool.plan_gen_expert.utils import train_utils
from models import networks
from torch.utils.tensorboard import SummaryWriter


class SimModel(nn.Module):
    """Wraps a model for simulation-based training with standard PyTorch."""

    def __init__(
        self,
        query_feat_dims: int,
        plan_feat_dims: int,
        torch_invert_cost: Optional[callable] = None,
        query_featurizer: Optional[Any] = None,
        perturb_query_features: Union[bool, Any] = False,
    ) -> None:
        super().__init__()
        # by default, it is using TreeConvolution
        self.tree_conv = treeconv.TreeConvolution(
            feature_size=query_feat_dims, plan_size=plan_feat_dims, label_size=1
        )

        self.torch_invert_cost = torch_invert_cost
        self.query_featurizer = query_featurizer
        self.perturb_query_features = perturb_query_features

    def forward(
        self,
        query_feat: torch.Tensor,
        plan_feat: torch.Tensor,
        indexes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.tree_conv(query_feat, plan_feat, indexes)

    def configure_optimizers(self) -> Tuple[torch.optim.Adam, None]:
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        return optimizer, None

    def compute_loss(
        self, batch: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        query_feat, plan_feat, *rest = batch
        target = rest[-1].to(BaseConfig.DEVICE)
        if self.training and self.perturb_query_features:
            query_feat = self.query_featurizer.PerturbQueryFeatures(
                query_feat, distribution=self.perturb_query_features
            )
        query_feat = query_feat.to(BaseConfig.DEVICE)
        plan_feat = plan_feat.to(BaseConfig.DEVICE)
        assert len(rest) == 2
        output = self.forward(query_feat, plan_feat, rest[0].to(BaseConfig.DEVICE))
        loss_value = F.mse_loss(
            output.reshape(
                -1,
            ),
            target.reshape(
                -1,
            ),
        )
        return loss_value

    def freeze(self):
        """Freeze all model parameters and set to eval mode (Lightning-style)."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        """Unfreeze all model parameters and set to train mode (Lightning-style)."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        params: Any,
        device: torch.device,
    ) -> None:
        """Custom training loop for SimModel."""
        optimizer, scheduler = self.configure_optimizers()
        self.to(BaseConfig.DEVICE)
        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0

        for epoch in range(params.epochs):
            self.train()
            train_loss_sum = 0.0
            train_batches = 0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.compute_loss(batch, device)
                loss.backward()
                if (
                    hasattr(params, "gradient_clip_val")
                    and params.gradient_clip_val > 0
                ):
                    torch.nn.utils.clip_grad_value_(
                        self.parameters(), params.gradient_clip_val
                    )
                if global_step % 50 == 0:
                    total_grad_norm = (
                        sum(
                            torch.norm(param.grad) ** 2
                            for param in self.parameters()
                            if param.grad is not None
                        )
                        ** 0.5
                    )
                    logging.info(
                        f"Step {global_step}: total_grad_norm={total_grad_norm:.4f}"
                    )
                optimizer.step()
                train_loss_sum += loss.item()
                train_batches += 1
                global_step += 1
                logging.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Train Loss: {loss.item():.4f}"
                )
            avg_train_loss = train_loss_sum / train_batches
            logging.info(f"Epoch {epoch + 1}, Average Train Loss: {avg_train_loss:.4f}")

            if val_loader is not None and params.validate_fraction > 0:
                self.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        val_loss = self.compute_loss(batch, device)
                        val_loss_sum += val_loss.item()
                        val_batches += 1
                avg_val_loss = val_loss_sum / val_batches
                logging.info(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(
                        {
                            "model_state_dict": self.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        "best_sim_model_checkpoint.pt",
                    )
                else:
                    patience_counter += 1
                    if (
                        hasattr(params, "validate_early_stop_patience")
                        and patience_counter >= params.validate_early_stop_patience
                    ):
                        logging.info(
                            f"Early stopping triggered after {patience_counter} validations with no improvement."
                        )
                        return


class SimModelBuilder:
    """Balsa simulation."""

    @classmethod
    def Params(cls) -> hyperparams.Params:
        p = hyperparams.InstantiableParams(cls)
        # Train
        p.define("epochs", 100, "Maximum training epochs. Early-stopping may kick in.")
        p.define(
            "gradient_clip_val",
            0,
            "Clip the gradient norm computed over all model parameters together. 0 means no clipping.",
        )
        p.define("bs", 2048, "Batch size.")

        # Validation
        p.define(
            "validate_fraction",
            0.1,
            "Sample this fraction of the dataset as the validation set. 0 to disable validation.",
        )
        # Search, train-time
        p.define(
            "search_params",
            DynamicProgramming.Params(),
            "Params of the enumeration routine to use for training data.",
        )
        # Search space
        p.define(
            "plan_physical",
            True,
            "Learn and plan physical scans/joins, or just join orders?",
        )

        # Infer, test-time
        # p.define('infer_search_method', 'beam_bk', 'Options: beam_bk.')
        # p.define('infer_beam_size', 20, 'Beam size.')
        # p.define('infer_search_until_n_complete_plans', 10, 'Search until how many complete plans?')

        # Workload
        # p.define('workload', envs.JoinOrderBenchmark.Params(), 'Params of the Workload, i.e., a set of queries.')
        # Data collection
        p.define(
            "generic_ops_only_for_min_card_cost",
            False,
            "If using MinCardCost, whether to enumerate generic ops only.",
        )
        p.define(
            "sim_data_collection_intermediate_goals",
            True,
            "For each query, also collect sim data with intermediate query goals?",
        )

        # Featurizations
        p.define(
            "plan_featurizer_cls",
            plan_graph_encoder.SimPlanFeaturizer,
            "Featurizer to use for plans.",
        )
        p.define(
            "query_featurizer_cls",
            query_graph_encoder.SimQueryFeaturizer,
            "Featurizer to use for queries.",
        )

        p.define("label_transforms", ["log1p", "standardize"], "Transforms for labels.")
        p.define("perturb_query_features", None, "See experiments.")

        # Eval
        p.define(
            "eval_output_path", "eval-cost.csv", "Path to write evaluation output into."
        )
        p.define(
            "eval_latency_output_path",
            "eval-latency.csv",
            "Path to write evaluation latency output into.",
        )

        return p

    @classmethod
    def hash_simulation_data(cls, p):
        """Gets the hash that should determine the simulation data."""
        # Use (a few attributes inside Params, Postgres configs) as hash key.
        # Using PG configs is necessary because things like PG version / PG
        # optimizer settings affect collected costs.
        # NOTE: in theory, other stateful effects such as whether ANALYZE has
        # been called on a PG database also affects the collected costs.
        _RELEVANT_HPARAMS = [
            "search_params",
            "generic_ops_only_for_min_card_cost",
            "plan_physical",
        ]
        param_vals = [p.get(hparam) for hparam in _RELEVANT_HPARAMS]
        param_vals = [
            v.to_text() if isinstance(v, hyperparams.Params) else str(v)
            for v in param_vals
        ]
        spec = "\n".join(param_vals)

        # if p.search.cost_model.cls is cost_model.PostgresCost:
        #     pg_configs = map(str, postgres.GetServerConfigs())
        #     spec += '\n'.join(pg_configs)
        hash_sim = hashlib.sha1(spec.encode()).hexdigest()[:8]
        return hash_sim

    @classmethod
    def hash_featurized_data(cls, p):
        """Gets the hash that should determine the final featurized tensors."""
        # Hash(HashOfSimData(), featurization specs).
        # NOTE: featurized data involves asking Postgres for cardinality
        # estimates of filters.  So in theory, here the hash calculation should
        # depend on postgres.GetServerConfigs().  Most relevant are the PG
        # version & whether ANALYZE has been run (this is not tracked by any PG
        # config).  Here let's make an assumption that all PG versions with
        # ANALYZE ran produce the same estimates, which is reasonable because
        # they are just histograms.
        hash_sim = cls.hash_simulation_data(p)
        _FEATURIZATION_HPARAMS = [
            "plan_featurizer_cls",
            "query_featurizer_cls",
        ]
        param_vals = [str(p.get(hparam)) for hparam in _FEATURIZATION_HPARAMS]
        spec = str(hash_sim) + "\n".join(param_vals)
        hash_feat = hashlib.sha1(spec.encode()).hexdigest()[:8]
        return hash_feat

    def __init__(
        self, params, workload_ins: workload.WorkloadInfo, db_cli: PostgresConnector
    ):
        # default config here
        self.skip_data_collection_geq_num_rels = 12

        self.cursor = db_cli

        self.params = params.copy()
        p = self.params
        logging.info(p)

        self.workload = workload_ins.Copy()
        self.all_query_nodes = self.workload.nodes
        self.train_nodes = self.all_query_nodes
        self.test_nodes = self.all_query_nodes

        p.search_params.plan_physical_ops = p.plan_physical
        p.search_params.cost_model.cost_physical_ops = p.plan_physical
        self.search: DynamicProgramming = p.search_params.cls(
            p.search_params, self.cursor
        )
        self.search.set_physical_ops(
            join_ops=self.workload.join_types, scan_ops=self.workload.scan_types
        )

        # this is only for testing sim
        if self.is_plan_physical_but_use_generic_ops():
            generic_join = ["Join"]
            generic_scan = ["Scan"]
            self.search.set_physical_ops(join_ops=generic_join, scan_ops=generic_scan)

        # A list of SubPlanTrainingPoint.
        self.simulation_data = []

        self.planner = None

        assert issubclass(
            p.plan_featurizer_cls, plan_graph_encoder.PhysicalTreeNodeFeaturizer
        )
        self.plan_featurizer = plan_graph_encoder.PhysicalTreeNodeFeaturizer(
            self.workload
        )
        assert issubclass(
            p.query_featurizer_cls, query_graph_encoder.SimQueryFeaturizer
        )
        self.query_featurizer = query_graph_encoder.SimQueryFeaturizer(self.workload)

        # training related data
        self.train_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.model = None

        logging.info(
            "{} train queries: {}".format(
                len(self.train_nodes),
                [node.info["query_name"] for node in self.train_nodes],
            )
        )
        logging.info(
            "{} test queries: {}".format(
                len(self.test_nodes),
                [node.info["query_name"] for node in self.test_nodes],
            )
        )

        workload.NodeOps.rewrite_as_generic_joinscan(self.all_query_nodes)

        # This call ensures that node.info['all_filters_est_rows'] is written, which is used by the query featurizer.
        # plan_gen_expert_exp.SimpleReplayBuffer(self.cursor, self.all_query_nodes)

    def _create_training_data_hook(
        self, accum: List, info_to_attach: Dict, num_rels: int
    ):
        """
        Factory method to create a hook function that collects training data points
        for query optimization during a dynamic programming (DP) trajectory.

        The hook labels each join subplan within a query plan with the total cost of
        the entire plan

        Args:
            self: Instance of the DynamicProgramming class.
            accum (List): Accumulator list to store SubPlanTrainingPoint objects.
            info_to_attach (Dict): Metadata (e.g., SQL string, query name) to attach to query nodes.
            num_rels (int): Number of tables in the original query.

        Returns:
            callable: A hook function that processes plans and costs, appending data points to `accum`.
        """
        p = self.params

        def Hook(plan: workload.Node, cost: float):
            if (
                not p.sim_data_collection_intermediate_goals
                and len(plan.GetLeaves()) < num_rels
            ):
                # Ablation: don't collect data on any plans/costs that have
                # fewer than 'num_rels' (the original query) tables.
                return

            query_node = plan.Copy()
            # NOTE: must make a copy as info can get new fields.
            query_node.info = dict(info_to_attach)
            query_node.cost = cost

            def _Helper(node):
                if node.IsJoin():
                    accum.append(
                        expert_datasets.SubPlanTrainingPoint(
                            subplan=node,
                            goal=query_node,
                            cost=cost,
                        )
                    )

            workload.NodeOps.map_node(query_node, _Helper)

        return Hook

    def _filter_lowest_cost_plans(self, points):
        """Deduplicates 'points' (assumed to be from the same query).

        For each unique (goal,subplan), keep the single datapoint with the best
        cost.  We need to check for smaller costs due to our data collection.

        Example:

            Enumerated plan: ((mc cn) t), say cost 100.
            Among all data points yielded from this plan, we will have:

                goal = {mc, cn, t}
                subplan = (mc cn)
                cost = 100

            However, the search procedure may enumerate another plan for the
            same goal, say (t (mc cn)) with cost 200.  Among all data points
            yielded from this plan, we will have:

                goal = {mc, cn, t}
                subplan = (mc cn)
                cost = 200

        So, we really want to keep the first, i.e., record only the cheapest
        for each unique (goal,subplan).
        """
        p = self.params
        best_cost = collections.defaultdict(lambda: np.inf)
        ret = {}
        for point in points:
            # NOTE: when this function turns the 'goal' part into a string,
            # some information is not preserved (e.g., the string doesn't
            # record filter info).  However, since we assume 'points' all come
            # from the same query, this simplification is OK for uniquifying.
            key = point.ToSubplanGoalHint(with_physical_hints=p.plan_physical)
            if point.cost < best_cost[key]:
                best_cost[key] = point.cost
                ret[key] = point
        return ret.values()

    def _sim_data_path(self):
        p = self.params
        hash_key = SimModelBuilder.hash_simulation_data(p)
        return "./sim_data.pkl"

        # if 'stack' in p.workload.query_dir:
        #     return 'data/sim-data-STACK-{}.pkl'.format(hash_key)
        # else:
        #     return 'data/sim-data-{}.pkl'.format(hash_key)

    def _load_sim_data(self):
        path = self._sim_data_path()
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.simulation_data = pickle.load(f)
        logging.info(
            "Loaded simulation data (len {}) from: {}".format(
                len(self.simulation_data), path
            )
        )
        logging.info(
            "Training data (first 10, total {}):".format(len(self.simulation_data))
        )
        logging.info("\n".join(map(str, self.simulation_data[:10])))
        return True

    def _save_sim_data(self):
        path = self._sim_data_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(self.simulation_data, f)
            logging.info(
                "Saved simulation data (len {}) to: {}".format(
                    len(self.simulation_data), path
                )
            )
        except Exception as e:
            logging.warning("Failed saving sim data:\n{}".format(e))

    def featurized_data_path(self):
        p = self.params
        hash_key = SimModelBuilder.hash_featurized_data(p)
        return "sim-featurized.pkl"
        # if 'stack' in p.workload.query_dir:
        #     return 'data/sim-featurized-STACK-{}.pkl'.format(hash_key)
        # else:
        #     return 'data/sim-featurized-{}.pkl'.format(hash_key)

    def _load_featurized_data(self):
        path = self.featurized_data_path()
        if not os.path.exists(path):
            return False, None
        with open(path, "rb") as f:
            data = torch.load(f)
        logging.info(
            "Loaded featurized data (len {}) from: {}".format(len(data[0]), path)
        )
        return True, data

    def _save_featurized_data(self, data):
        path = self.featurized_data_path()
        try:
            with open(path, "wb") as f:
                torch.save(data, f)
            logging.info(
                "Saved featurized data (len {}) to: {}".format(len(data[0]), path)
            )
        except Exception as e:
            logging.warning("Failed saving featurized data:\n{}".format(e))

    def collect_sim_data(self):
        if self._load_sim_data():
            return

        start = time.time()
        num_collected = 0
        for query_node in self.train_nodes:
            num_rels = len(query_node.leaf_ids())
            logging.info(
                "query={} num_rels={}".format(query_node.info["query_name"], num_rels)
            )

            if num_rels >= self.skip_data_collection_geq_num_rels:
                continue
            num_collected += 1

            # Accumulate data points from this query.
            accum = []
            info_to_attach = {
                "overall_join_graph": query_node.info["parsed_join_graph"],
                "overall_join_conds": query_node.info["parsed_join_conds"],
                "path": query_node.info["path"],
            }

            self.search.push_on_enumerated_hook(
                self._create_training_data_hook(accum, info_to_attach, num_rels)
            )
            _, _, explored_sql_num = self.search.collect_for_single_query(query_node)

            print(
                f"[collect_sim_data]: have explored {explored_sql_num} sub plans for {query_node.info['query_name']}"
            )

            self.search.pop_on_enumerated_hook()
            new_accum = self._filter_lowest_cost_plans(accum)
            self.simulation_data.extend(new_accum)
            logging.info(
                "{} points before uniquifying, {} after".format(
                    len(accum), len(new_accum)
                )
            )

        # Gather filter info and estimate selectivity for all nodes
        explored_full_plans = [ele.goal for ele in self.simulation_data]
        for plan in explored_full_plans:
            plan.gather_filter_info()
        plan_node.PGToNodeHelper.estimate_nodes_selectivility(
            self.cursor, explored_full_plans
        )

        simulation_time = time.time() - start

        logging.info("Collection done, stats:")
        logging.info(
            " num_queries={} num_collected_queries={} num_points={}"
            " latency_s={:.1f}".format(
                len(self.train_nodes),
                num_collected,
                len(self.simulation_data),
                simulation_time,
            )
        )
        self._save_sim_data()
        return simulation_time, len(self.simulation_data)

    def _make_model(self, query_feat_dims: int, plan_feat_dims: int):
        p = self.params

        logging.info(
            "SIM query_feat_dims={} plan_feat_dims={}".format(
                query_feat_dims, plan_feat_dims
            )
        )
        logging.info(
            "SIM query_feat={} plan_feat={}".format(
                p.query_featurizer_cls, p.plan_featurizer_cls
            )
        )
        model = SimModel(
            query_feat_dims=query_feat_dims,
            plan_feat_dims=plan_feat_dims,
            torch_invert_cost=self.train_dataset.dataset.TorchInvertCost,
            query_featurizer=self.query_featurizer,
            perturb_query_features=p.perturb_query_features,
        )
        train_utils.report_model(model)
        return model

    def data_preprocessing(self):
        """Pre-processes/featurizes simulation data into tensors."""

        done, data = self._load_featurized_data()
        if done:
            return data

        logging.info("Creating SimpleReplayBuffer")
        # The constructor of SRB realy only needs goal/query Nodes for
        # instantiating workload info metadata and featurizers (e.g., grab all table names).

        explored_full_plans = [p.goal for p in self.simulation_data]
        explored_sub_plans = [p.subplan for p in self.simulation_data]

        logging.info("featurize_with_subplans()")
        t1 = time.time()
        all_query_vecs = [self.query_featurizer(node) for node in explored_full_plans]
        all_costs = [node.cost for node in explored_full_plans]
        print("Spent {:.1f}s on query and cost featurization".format(time.time() - t1))
        all_plan_feat_vecs, all_plan_pos_vecs = (
            plan_graph_encoder.make_and_featurize_trees(
                self.plan_featurizer, explored_sub_plans
            )
        )

        # Debug print
        for i in range(min(len(explored_full_plans), 10)):
            print(
                "query={} plan={} cost={}".format(
                    (
                        all_query_vecs[i] * np.arange(1, 1 + len(all_query_vecs[i]))
                    ).sum(),
                    all_plan_feat_vecs[i],
                    all_costs[i],
                )
            )

        data = (all_query_vecs, all_plan_feat_vecs, all_plan_pos_vecs, all_costs)

        self._save_featurized_data(data)
        return data

    def train(self, load_from_checkpoint=None, log_dir: str = "logs"):
        """
        Trains the model using PyTorch, processing and featurizing training data,
        initializing the model, and running the training loop.

        Args:
            load_from_checkpoint (str, optional): Path to a checkpoint file to load model weights.
            log_dir (str): Directory for TensorBoard logs (default: "logs").

        Returns:
            The processed training data.
        """
        p = self.params

        # Pre-process and featurize data
        data = self.data_preprocessing()

        # Create DataLoader
        logging.info("Creating Dataset and DataLoader")
        (
            self.train_dataset,
            self.train_loader,
            _,
            self.val_loader,
            num_train,
            num_validation,
        ) = expert_datasets.graphexp_make_dataloader(
            data=data,
            label_transforms=p.label_transforms,
            validate_fraction=p.validate_fraction,
            batch_size=p.bs,
        )
        logging.info("num_train={} num_validation={}".format(num_train, num_validation))

        batch = next(iter(self.train_loader))
        # logging.info('Example batch (query, plan, indexes, cost):\n{}'.format(batch))

        # Initialize model
        _, query_feat_dims = batch[0].shape
        if issubclass(
            p.plan_featurizer_cls, plan_graph_encoder.TreeNodeFeaturizer
        ) or issubclass(
            p.plan_featurizer_cls, plan_graph_encoder.PhysicalTreeNodeFeaturizer
        ):
            unused_bs, plan_feat_dims, unused_max_tree_nodes = batch[1].shape
            logging.info(
                "Batch shape: (batch_size, plan_feat_dims, max_tree_nodes) = {}".format(
                    (unused_bs, plan_feat_dims, unused_max_tree_nodes)
                )
            )
        else:
            unused_bs, plan_feat_dims = batch[1].shape
        self.model = self._make_model(
            query_feat_dims=query_feat_dims, plan_feat_dims=plan_feat_dims
        )
        self.model.to(BaseConfig.DEVICE)

        # Initialize TensorBoard writer, Log hyperparameters
        writer = SummaryWriter(log_dir=os.path.join(log_dir, "training"))
        p_dict = hyperparams.sanitize_to_text(dict(p))
        writer.add_hparams(p_dict, {})

        # Train or load from checkpoint
        if os.path.exists(load_from_checkpoint):
            logging.info(f"Loading pretrained checkpoint: {load_from_checkpoint}")
            self.model.load_state_dict(torch.load(load_from_checkpoint))
        else:
            logging.info("Starting training")
            self._train_model(writer)

        # writer.close()
        # return data

    def _train_model(self, writer: SummaryWriter):
        """
        Trains the model using a custom PyTorch training loop with early stopping.
        """
        p = self.params

        # Initialize optimizer (assuming Adam as a default, adjust as needed)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=p.learning_rate if hasattr(p, "learning_rate") else 0.001,
        )

        # Initialize learning rate scheduler (optional, e.g., ReduceLROnPlateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, verbose=True
        )

        # Early stopping parameters
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0
        best_model_path = os.path.join(os.getcwd(), "best_model.pth")

        # Training loop
        for epoch in range(p.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                query, plan, indexes, cost = [x.to(BaseConfig.DEVICE) for x in batch]

                # Forward pass
                optimizer.zero_grad()
                output = self.model(
                    query, plan, indexes
                )  # Adjust based on actual model signature
                loss = nn.MSELoss()(output, cost)

                # Backward pass and optimize
                loss.backward()
                if hasattr(p, "gradient_clip_val") and p.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), p.gradient_clip_val
                    )
                optimizer.step()

                train_loss += loss.item()
                if batch_idx % 10 == 0:  # Log every 10 batches (row_log_interval)
                    writer.add_scalar(
                        "Loss/Train",
                        loss.item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )

            # Average training loss
            train_loss /= len(self.train_loader)
            writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)

            # Validation phase
            if p.validate_fraction > 0 and self.val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in self.val_loader:
                        query, plan, indexes, cost = [
                            x.to(BaseConfig.DEVICE) for x in batch
                        ]
                        output = self.model(query, plan, indexes)
                        val_loss += nn.MSELoss()(output, cost).item()
                    val_loss /= len(self.val_loader)
                    writer.add_scalar("Loss/Validation", val_loss, epoch)

                # Update scheduler
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), best_model_path)
                    logging.info(
                        f"Saved best model with validation loss: {best_val_loss:.4f}"
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(
                            f"Early stopping at epoch {epoch + 1}, best validation loss: {best_val_loss:.4f}"
                        )
                        break

            logging.info(
                f"Epoch {epoch + 1}/{p.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        # Load best model
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            logging.info(f"Loaded best model from {best_model_path}")

    def inference(self, node):
        """Runs query planning on 'node'.

        Returns:
          plan: the found plan.
          cost: the learned, inverse-transformed cost of the found plan.
        """
        p = self.params
        planner = self._get_optimizer()
        # bushy = True
        # if planner_config is not None:
        #     bushy = planner_config.search_space == 'bushy'
        planniong_time, plan, cost, _ = planner.plan(node)
        print(f"planning for this sql requires {planniong_time} time usage")
        return plan, cost

    def evaluate_cost(self):
        """Reports cost sub-optimalities w.r.t. Postgres."""
        p = self.params
        metrics = []
        qnames = []
        num_rels = []
        # todo: this is already loaded, is it?
        # self._load_best_checkpoint()

        if not isinstance(self.search.cost_model, cost_model.PostgresCost):
            return

        for query_node in self.all_query_nodes:
            qnames.append(query_node.info["query_name"])
            num_rels.append(len(query_node.leaf_ids()))

            # Call FilterScanOrJoins so that things like Aggregate/Hash are removed and don't add parentheses in the hint string.
            pg_plan_str = query_node.filter_scans_joins().hint_str()
            logging.info("query={} num_rels={}".format(qnames[-1], num_rels[-1]))
            logging.info(
                "postgres_plan={} postgres_cost={}".format(pg_plan_str, query_node.cost)
            )

            found_plan, predicted_cost = self.inference(query_node)

            if isinstance(self.search.cost_model, cost_model.PostgresCost):
                # Score via PG.
                # Due to quirkiness in PG (e.g., different FROM orderings produce
                # different costs, even though the join orders are exactly the
                # same), we use the original SQL query here with 'found_plan''s
                # hint_str().  Doing this makes sure suboptimality is at best 1.0x
                # for non-GEQO plans.
                cost_of_found_plan = self.search.cost_model.score_with_sql(
                    found_plan, query_node.info["sql_str"]
                )
                found_plan_str = found_plan.hint_str()
                cost_of_pg_plan = query_node.cost

                suboptimality = cost_of_found_plan / cost_of_pg_plan

                # Sanity checks.
                # if worse
                if suboptimality > 1:
                    assert pg_plan_str != found_plan_str, (pg_plan_str, found_plan_str)
                # if better
                elif suboptimality < 0.99:
                    # Check that we can only do better than PG plan under the
                    # cost model if GEQO is enabled.  Otherwise PG uses
                    # exhaustive DP so should be optimal.
                    #
                    # We use 0.99 because e.g., q13a, even the query plans are
                    # the same w/ and w/o hinting, a top Gather node can have
                    # slightly different costs, on the order of ~0.2).
                    GEQO_THRESHOLD = 12
                    assert num_rels[-1] >= GEQO_THRESHOLD, num_rels[-1]

            else:
                cost_of_found_plan = predicted_cost
                suboptimality = 1

            # Logging.
            metrics.append(suboptimality)
            logging.info("  predicted_cost={:.1f}".format(predicted_cost))
            logging.info("  actual_cost={:.1f}".format(cost_of_found_plan))
            logging.info("  suboptimality={:.1f}".format(suboptimality))

        df = pd.DataFrame(
            {"query": qnames, "num_rel": num_rels, "suboptimality": metrics}
        )
        df.to_csv(p.eval_output_path)
        logging.info("suboptimalities:\n{}".format(df["suboptimality"].describe()))

    def evaluate_latency(self, planner_config=None):
        p = self.params
        metrics = []
        qnames = []
        num_rels = []

        for query_node in self.all_query_nodes:
            qnames.append(query_node.info["query_name"])
            num_rels.append(len(query_node.leaf_ids()))

            # Call FilterScanOrJoins so that things like Aggregate/Hash are
            # removed and don't add parentheses in the hint string.
            pg_plan_str = query_node.filter_scans_joins().hint_str()
            logging.info("query={} num_rels={}".format(qnames[-1], num_rels[-1]))
            logging.info(
                "postgres_plan={} postgres_cost={}".format(pg_plan_str, query_node.cost)
            )

            found_plan, predicted_cost = self.inference(query_node)
            actual_cost = plan_node.PGToNodeHelper.get_real_pg_latency(
                db_cli=self.cursor,
                sql=query_node.info["sql_str"],
                hint=found_plan.hint_str(p.plan_physical),
            )

            # Logging.
            metrics.append(actual_cost)
            logging.info("  predicted_cost={:.1f}".format(predicted_cost))
            logging.info("  actual_latency_ms={:.1f}".format(actual_cost))

        df = pd.DataFrame(
            {
                "query": qnames,
                "num_rel": num_rels,
                "latency": metrics,
            }
        )
        df.to_csv(p.eval_latency_output_path)
        # self.trainer.logger.log_metrics({
        #     'latency_sum_s': np.sum(metrics) / 1e3,
        # })
        logging.info("Latencies:\n{}".format(df["latency"].describe()))

    def predict(self, query_node, nodes):
        """Runs forward pass on 'nodes' to predict their costs."""
        return self._get_optimizer().infer(query_node, nodes)

    def _load_best_checkpoint(self, checkpoint_path: str = None):
        """
        Loads the model state from the best checkpoint based on validation loss.
        Args:
            checkpoint_path (Optional[str]): Path to the checkpoint file. If None,
                defaults to 'best_model.pth' in the current directory.
        Returns:
            bool: True if the checkpoint was loaded successfully, False otherwise.
        """
        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.getcwd(), "best_model.pth")

        if os.path.exists(checkpoint_path):
            # Calculate sum of model parameters before loading (for debugging)
            old_sum = sum(
                weight.sum().item() for _, weight in self.model.named_parameters()
            )

            # Load checkpoint
            try:
                state_dict = torch.load(
                    checkpoint_path, map_location=torch.device("cpu")
                )
                # Handle both direct state_dict and Lightning-style checkpoint
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict)

                # Calculate sum of model parameters after loading
                new_sum = sum(
                    weight.sum().item() for _, weight in self.model.named_parameters()
                )
                logging.info(
                    f"Loaded best checkpoint from {checkpoint_path}, param sum: {old_sum:.4f} -> {new_sum:.4f}"
                )
                return True
            except Exception as e:
                logging.warning(
                    f"Failed to load checkpoint from {checkpoint_path}: {e}"
                )
                return False
        else:
            logging.warning(
                f"No checkpoint found at {checkpoint_path}; model unchanged."
            )
            return False

    def free_resources(self):
        self.simulation_data = None
        self.train_loader = None
        self.val_loader = None
        self.train_dataset.dataset.FreeData()

    def _get_optimizer(self):
        p = self.params
        if self.planner is None:
            wi = self.workload
            wi_for_ops_to_enum = wi
            if self.is_plan_physical_but_use_generic_ops():
                wi_for_ops_to_enum = copy.deepcopy(wi)
                # We want to make sure the Optimizer enumerates just physical
                # ops (no generic ops), but still being able to use
                # query_featurizer/plan_featurizer that knows about both
                # physical+generic ops.
                wi_for_ops_to_enum.all_ops = np.asarray(
                    [
                        op
                        for op in wi_for_ops_to_enum.all_ops
                        if op not in ["Join", "Scan"]
                    ]
                )
                wi_for_ops_to_enum.join_types = np.asarray(
                    [op for op in wi_for_ops_to_enum.join_types if op != "Join"]
                )
                wi_for_ops_to_enum.scan_types = np.asarray(
                    [op for op in wi_for_ops_to_enum.scan_types if op != "Scan"]
                )
            self.planner = BMOptimizer(
                workload_info=wi_for_ops_to_enum,
                plan_featurizer=p.plan_featurizer_cls(wi),
                parent_pos_featurizer=None,  # parent_pos_featurizer
                query_featurizer=self.query_featurizer,
                inverse_label_transform_fn=self.train_dataset.dataset.InvertCost,
                model=self.model,
                plan_physical=p.plan_physical,
            )
        else:
            # Otherwise, 'self.model' might have been updated since the planner is created.  Update it.
            self.planner.set_model_for_eval(self.model)
        return self.planner

    def is_plan_physical_but_use_generic_ops(self):
        p = self.params
        # This is a logical-only cost model.  Let's only enumerate generic ops.
        return (
            p.plan_physical
            and p.generic_ops_only_for_min_card_cost
            and isinstance(self.search.cost_model, cost_model.MinCardCost)
        )


def Main(argv):
    # Ignore argv as it's unused
    del argv
    from expert_pool.plan_gen_expert import GraphOptimizerExpert

    p = SimModelBuilder.Params()

    p.generic_ops_only_for_min_card_cost = True

    p.search_params.cost_model = cost_model.MinCardCost.Params()

    p.plan_featurizer_cls = plan_graph_encoder.TreeNodeFeaturizer

    # Infer.
    p.infer_beam_size = 20
    p.infer_search_until_n_complete_plans = 10

    p.plan_physical = True
    if p.plan_physical:
        p.plan_featurizer_cls = plan_graph_encoder.PhysicalTreeNodeFeaturizer

    # Pre-training via simulation data.
    with PostgresConnector("imdb_ori") as conn:
        workload_ins = GraphOptimizerExpert.workload_builder(
            db_cli=conn,
            input_sql_dir="/Users/kevin/project_python/AI4QueryOptimizer/experiment_setup/workloads/bao/join_unique_mini",
            init_baseline_exp=p.init_baseline_experience,
        )
        sim = SimModelBuilder(params=p, db_cli=conn, workload_ins=workload_ins)
        sim.collect_sim_data()
        # Use None to retrain; pass a ckpt to reload.
        sim_ckpt = None
        train_data = None
        for i in range(5):
            train_data = sim.train(train_data, load_from_checkpoint=sim_ckpt)
            sim.params.eval_output_path = "eval-cost-{}.csv".format(i)
            sim.params.eval_latency_output_path = "eval-latency-{}.csv".format(i)
            sim.evaluate_cost()
            sim.evaluate_latency()


if __name__ == "__main__":
    app.run(Main)
