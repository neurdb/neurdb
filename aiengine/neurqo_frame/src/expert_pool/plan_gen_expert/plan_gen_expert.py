import collections
import copy
import datetime
import json
import os
import pickle
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from common import workload, config_imdb
from db.pg_conn import PostgresConnector
from parser import plan_node
from expert_pool.plan_gen_expert.utils import io, train_utils

# Import from local package
from expert_pool.plan_gen_expert import plan_gen_expert_exp, plan_gen_expert_sim
from expert_pool.plan_gen_expert.encoders import plan_graph_encoder
from expert_pool.plan_gen_expert.encoders import sql_graph_encoder as query_graph_encoder
from expert_pool.plan_gen_expert.models import cost_model, treeconv
from expert_pool.plan_gen_expert.dataset import expert_datasets
from expert_pool.plan_gen_expert.search_alg.beam import BMOptimizer

device = "cuda" if torch.cuda.is_available() else "cpu"


class PerQueryTimeoutController:
    """Bounds the total duration of an iteration over the workload."""

    # all thise are default, don't set up further.
    def __init__(self,
                 timeout_slack=2,
                 no_op=False,
                 relax_timeout_factor=None,
                 relax_timeout_on_n_timeout_iters=None,
                 initial_timeout_ms=None):
        self.timeout_slack = timeout_slack
        self.no_op = no_op
        self.relax_timeout_factor = relax_timeout_factor
        self.relax_timeout_on_n_timeout_iters = relax_timeout_on_n_timeout_iters
        # Fields to maintain.
        self.iter_timeout_ms = initial_timeout_ms
        self.curr_iter_ms = None
        self.curr_iter_max_ms = None
        self.curr_iter_has_timeouts = False
        self.num_consecutive_timeout_iters = 0
        self.iter_executed = False

    def GetTimeout(self, query_node):
        if self.no_op:
            return None
        if self.iter_timeout_ms is None:
            return None
        # If all queries in the previous iteration time out, iter_timeout_ms
        # would be -1e30; guard against this by returning 0 (no timeout).
        return max(self.iter_timeout_ms, 0)

    def RecordQueryExecution(self, query_node, latency_ms):
        del query_node
        if self.no_op:
            return
        self.iter_executed = True
        if latency_ms < 0:
            # This query timed out.
            self.curr_iter_has_timeouts = True
        else:
            # This query finished within timeout.
            self.curr_iter_ms += latency_ms
            self.curr_iter_max_ms = max(self.curr_iter_max_ms, latency_ms)

    def OnIterStart(self):
        if self.no_op:
            return
        # NOTE: Suppose we call OnIterStart() then due to errors do not call
        # any RecordQueryExecution().  Due to retries we call OnIterStart()
        # again.  At this point, curr_iter_max_ms = -1e30 and iter_timeout_ms =
        # None and iter_executed = False, and we should not update the timeout.
        if self.curr_iter_max_ms is not None and self.iter_executed:
            # Update timeout.
            if self.iter_timeout_ms is None:
                self.iter_timeout_ms = \
                    self.curr_iter_max_ms * self.timeout_slack
            elif not self.curr_iter_has_timeouts:
                self.iter_timeout_ms = min(
                    self.iter_timeout_ms,
                    self.curr_iter_max_ms * self.timeout_slack)

            # How long have we been consecutively timing out up to now?
            if self.curr_iter_has_timeouts:
                self.num_consecutive_timeout_iters += 1
            else:
                self.num_consecutive_timeout_iters = 0

            # Optionally, relax the timeout.
            if (self.relax_timeout_factor is not None and
                    self.num_consecutive_timeout_iters >=
                    self.relax_timeout_on_n_timeout_iters):
                self.iter_timeout_ms *= self.relax_timeout_factor

        self.curr_iter_ms = 0
        self.curr_iter_max_ms = -1e30
        self.curr_iter_has_timeouts = False
        self.iter_executed = False


class QueryExecutionCache:
    """A simple cache mapping key -> (best value, best latency).
    To record (best result, best latency) per (query name, plan):
        # Maps key to (best value, best latency).
        Put(key=(query_name, hint_str), value=result_tup, latency=latency)
    To record (best Node, best latency) per query name:
        Put(key=query_name, value=node, latency=latency)
    """

    def __init__(self):
        self._cache = {}
        self._counts = {}

    def size(self):
        return len(self._cache)

    def Put(self, key, value, latency):
        """Put.

        Updates key -> (value, latency), iff
        (1) no existing value is found or
        (2) latency < the existing latency.

        Args:
          key: the key.  E.g., (query_name, hint_str) which identifies a unique
            query plan.
          value: the value.  E.g., a ResultTup or a Node.
          latency: the latency.
        """
        prior_result = self._cache.get(key)
        if prior_result is None:
            prior_latency = np.inf
        else:
            prior_latency = prior_result[1]
        if latency < prior_latency:
            self._cache[key] = (value, latency)
        # Update visit counts.
        cnt = self.GetVisitCount(key)
        self._counts[key] = cnt + 1

    def Get(self, key):
        return self._cache.get(key)

    def GetVisitCount(self, key):
        return self._counts.get(key, 0)


class BalsaModel(nn.Module):
    """Vanilla PyTorch version of BalsaModel (no Lightning)."""

    def __init__(self,
                 params: Any,
                 model: nn.Module,
                 loss_type: Optional[str] = None,
                 torch_invert_cost: Optional[callable] = None,
                 query_featurizer: Optional[Any] = None,
                 perturb_query_features: Optional[Any] = None,
                 l2_lambda: float = 0,
                 learning_rate: float = 1e-3,
                 optimizer_state_dict: Optional[Dict] = None,
                 reduce_lr_within_val_iter: bool = False) -> None:
        super().__init__()
        self.params = params.Copy() if hasattr(params, "Copy") else params
        self.model = model
        assert loss_type in [None, 'mean_qerror'], loss_type
        self.loss_type = loss_type
        self.torch_invert_cost = torch_invert_cost
        self.query_featurizer = query_featurizer
        self.perturb_query_features = perturb_query_features
        self.l2_lambda = l2_lambda
        self.optimizer_state_dict = optimizer_state_dict
        self.learning_rate = learning_rate
        self.reduce_lr_within_val_iter = reduce_lr_within_val_iter

    def forward(self,
                query_feat: torch.Tensor,
                plan_feat: torch.Tensor,
                indexes: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        return self.model(query_feat, plan_feat, indexes)

    def compute_loss(self,
                     batch: Any,
                     training: bool = True
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        p = self.params
        query_feat = batch.query_feats.to(device)

        if training and self.perturb_query_features is not None:
            query_feat = self.query_featurizer.PerturbQueryFeatures(
                query_feat, distribution=self.perturb_query_features
            )

        query_feat, plan_feat, indexes, target = (
            query_feat,
            batch.plans.to(device),
            batch.indexes.to(device),
            batch.costs.to(device),
        )

        output = self.forward(query_feat, plan_feat, indexes)

        if p.cross_entropy:
            log_probs = output.log_softmax(-1)
            target_dist = torch.zeros_like(log_probs)
            # Scalar 46.25 represented as: 0.75 * 46 + 0.25 * 47.
            ceil = torch.ceil(target)
            w_ceil = ceil - target
            floor = torch.floor(target)
            w_floor = 1 - w_ceil
            target_dist.scatter_(1,
                                 ceil.long().unsqueeze(1), w_ceil.unsqueeze(1))
            target_dist.scatter_(1,
                                 floor.long().unsqueeze(1),
                                 w_floor.unsqueeze(1))
            loss = (-target_dist * log_probs).sum(-1).mean()
        else:
            # this is default
            loss = F.mse_loss(output.reshape(-1, ), target.reshape(-1, ))
            # if self.loss_type == "mean_qerror":
            #     output_inverted = self.torch_invert_cost(output.reshape(-1, ))
            #     target_inverted = self.torch_invert_cost(target.reshape(-1, ))
            #     loss = train_utils.QErrorLoss(output_inverted, target_inverted)
            # else:
            #     loss = F.mse_loss(output.reshape(-1, ), target.reshape(-1, ))

        l2_loss = None
        if self.l2_lambda > 0:
            l2_loss = torch.tensor(0., device=device, requires_grad=True)
            for param in self.parameters():
                l2_loss += torch.norm(param).pow(2)
            l2_loss = self.l2_lambda * 0.5 * l2_loss
            loss = loss + l2_loss

        return loss, l2_loss

    def make_optimizer(self):
        if self.params.adamw:
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.params.adamw,
            )
        else:
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.optimizer_state_dict is not None:
            curr = opt.state_dict()['param_groups'][0]['params']
            prev = self.optimizer_state_dict['param_groups'][0]['params']
            assert curr == prev, (curr, prev)
            print('Loading last iter\'s optimizer state.')
            # Prev optimizer state's LR may be stale.
            opt.load_state_dict(self.optimizer_state_dict)
            for param_group in opt.param_groups:
                param_group['lr'] = self.learning_rate
            assert opt.state_dict()['param_groups'][0]['lr'] == self.learning_rate
            print('LR', self.learning_rate)
        return opt

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, num_epochs: int = 10):
        """Run a simple training loop, and restore best checkpoint (like Lightning)."""

        self.to(device)
        optimizer = self.make_optimizer()
        scheduler = None
        if self.reduce_lr_within_val_iter:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, verbose=True
            )

        best_val_loss = float("inf")
        best_state_dict = None

        for epoch in range(1, num_epochs + 1):
            # --- Training ---
            self.train()
            train_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                loss, _ = self.compute_loss(batch, training=True)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)

            # --- Validation ---
            avg_val_loss = None
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        val_loss, _ = self.compute_loss(batch, training=False)
                        val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)

                # track best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state_dict = self.state_dict()

                if scheduler:
                    scheduler.step(avg_val_loss)

            # --- Logging ---
            if avg_val_loss is not None:
                print(
                    f"[Epoch {epoch:03d}] Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}"
                )
            else:
                print(f"[Epoch {epoch:03d}] Train loss: {avg_train_loss:.4f}")

        # --- Restore best checkpoint ---
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            print(f"Restored best checkpoint (val_loss={best_val_loss:.4f})")


class GraphOptimizerExpert:
    search_space_join_ops = ['Hash Join', 'Merge Join', 'Nested Loop']
    search_space_scan_ops = ['Index Scan', 'Index Only Scan', 'Seq Scan', 'Bitmap Heap Scan', 'Tid Scan'],

    def __init__(self, params, db_cli: PostgresConnector = None):

        # default value here

        super().__init__()
        self.time_temp = datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')
        self.db_cli = db_cli
        self.params = params.copy()

        self.workload, run_baseline = GraphOptimizerExpert.workload_builder(
            db_cli=self.db_cli,
            input_sql_dir="/Users/kevin/project_python/AI4QueryOptimizer/experiment_setup/workloads/bao/join_unique_mini",
            init_baseline_exp=self.params.init_baseline_experience)
        self.train_nodes: List[workload.Node] = self.workload.nodes
        self.test_nodes: List[workload.Node] = self.workload.nodes

        self.params.run_baseline = run_baseline

        self.ema_source_net = None

        self.timeout_controller = PerQueryTimeoutController()
        self.query_execution_cache = QueryExecutionCache()
        self.best_plans = QueryExecutionCache()

        # Labels.
        self.label_mean = None
        self.label_std = None
        self.label_running_stats = train_utils.RunningStats()

        # EMA/SWA.
        #   average name -> dict
        self.moving_average_model_state_dict = collections.defaultdict(dict)
        #   average name -> counter
        self.moving_average_counter_dict = collections.defaultdict(int)

        # LR schedule.
        self.lr_schedule = train_utils.GetLrSchedule(self.params)

        # Optimizer state.
        self.prev_optimizer_state_dict = None

        self.timer = train_utils.Timer()

        self.sim = None
        self.exp = None
        self.exp_val = None

        self._latest_replay_buffer_path = None

        self.summary_writer = SummaryWriter()

        # training-related metrics
        self.curr_value_iter = 0
        self.num_query_execs = 0
        self.num_total_timeouts = 0
        self.curr_iter_skipped_queries = 0
        self.overall_best_train_latency = np.inf
        self.overall_best_test_latency = np.inf
        self.overall_best_test_swa_latency = np.inf
        self.overall_best_test_ema_latency = np.inf

    def _make_exp_buffer(self):

        # Use the already instantiated query featurizer, which may contain computed normalization stats.
        exp = plan_gen_expert_exp.Experience(
            db_cli=self.db_cli,
            data=self.train_nodes,
            workload_info=self.workload,
            query_featurizer_cls=self.sim.query_featurizer.__class__,
            query_featurizer=self.sim.query_featurizer,
            plan_featurizer_cls=plan_graph_encoder.PhysicalTreeNodeFeaturizer)

        # by default, this is None.
        exp_val = None
        return exp, exp_val

    @staticmethod
    def workload_builder(db_cli: PostgresConnector, input_sql_dir: str, init_baseline_exp: str):
        search_space_join_ops = ['Hash Join', 'Merge Join', 'Nested Loop']
        search_space_scan_ops = ['Index Scan', 'Index Only Scan', 'Seq Scan', 'Bitmap Heap Scan', 'Tid Scan']

        if os.path.isfile(init_baseline_exp):
            print("Loading the workload infor from a file")
            with open(init_baseline_exp, 'rb') as f:
                workload_ins = pickle.load(f)

            need_run_baseline = False
        else:
            query_w_name = read_sql_files(input_sql_dir)
            all_query_nodes = []
            for sql_file_name, sql_str in query_w_name:
                query_node = plan_node.PGToNodeHelper.sql_to_plan_node_w_parsed_sql(db_cli, sql_str, sql_file_name)
                all_query_nodes.append(query_node)

            workload_ins = workload.WorkloadInfo(all_query_nodes)
            workload_ins.SetPhysicalOps(search_space_join_ops, search_space_scan_ops)
            workload_ins.table_num_rows = config_imdb.GetAllTableNumRows(workload_ins.rel_names)

            need_run_baseline = True

        return workload_ins, need_run_baseline

    def get_train_sim_model(self):
        if self.sim is not None:
            return self.sim

        p = self.params

        sim_p = plan_gen_expert_sim.SimModelBuilder.Params()
        # # Copy over relevant params
        # if 'stack' in p.query_dir:
        #     sim_p.workload = envs.STACK.Params()
        # else:
        # sim_p.workload.query_dir = p.query_dir
        # sim_p.workload.query_glob = p.query_glob
        # sim_p.workload.test_query_glob = p.test_query_glob
        # sim_p.workload.search_space_join_ops = p.search_space_join_ops
        # sim_p.workload.search_space_scan_ops = p.search_space_scan_ops

        if p.cost_model == 'mincardcost':
            sim_p.search_params.cost_model = cost_model.MinCardCost.Params()
        else:
            sim_p.search_params.cost_model = cost_model.PostgresCost.Params()

        sim_p.query_featurizer_cls = query_graph_encoder.SimQueryFeaturizer
        sim_p.plan_featurizer_cls = plan_graph_encoder.TreeNodeFeaturizer

        if p.plan_physical:
            sim_p.plan_physical = True
            sim_p.plan_featurizer_cls = plan_graph_encoder.PhysicalTreeNodeFeaturizer

        sim_p.generic_ops_only_for_min_card_cost = p.generic_ops_only_for_min_card_cost
        sim_p.label_transforms = p.label_transforms

        sim_p.gradient_clip_val = p.gradient_clip_val
        sim_p.bs = p.bs
        sim_p.epochs = p.epochs
        sim_p.perturb_query_features = p.perturb_query_features
        sim_p.validate_fraction = p.validate_fraction

        # Instantiate.
        sim = plan_gen_expert_sim.SimModelBuilder(params=sim_p, db_cli=self.db_cli, workload_ins=self.workload)
        sim.collect_sim_data()
        sim.train(load_from_checkpoint=p.sim_checkpoint)
        sim.model.freeze()
        # sim.evaluate_cost()
        sim.free_resources()
        return sim

    def run_baseline_optimzier(self):
        """
        Update cost with real-cost, and save the updated experience.
        """
        p = self.params

        self.db_cli.drop_buffer_cache()

        # update the cost with the real latecy
        for node in self.workload.nodes:
            query_tree, _ = self.db_cli.explain_analysis_read_sql(
                query="explain (verbose, analyze, format json)" + '\n' + node.info['sql_str'],
                geqo_off=True)

            real_cost = query_tree['Execution Time']
            plan_time = query_tree['Planning Time']
            node.cost = real_cost
            node.info['explain_json'] = query_tree
            print(f'q{node.info["query_name"]},{real_cost:.1f} (baseline), '
                  f'Execution time = {real_cost}, Planning Time = {plan_time}')

        # NOTE: if engine != pg, we're still saving PG plans but with target
        # engine's latencies.  This mainly affects debug strings.
        if 'stack' in p.query_dir:
            io.save_pickle(self.workload, './initial_policy_data__stack.pkl')
        else:
            io.save_pickle(self.workload, './initial_policy_data.pkl')
        self.LogExpertExperience(self.train_nodes, self.test_nodes)

    def _MakeModel(self, dataset, train_from_scratch=False):
        p = self.params
        if not hasattr(self, 'model') or p.skip_sim_init_iter_1p:
            # Init afresh if either the model has not been constructed, or if
            # 'p.skip_sim_init_iter_1p', which explicitly says we want a fresh
            # model on iters >= 1.
            print('MakeModel afresh')
            num_label_bins = int(dataset.costs.max().item()) + 2  # +1 for 0, +1 for ceil(max cost).
            query_feat_size = len(self.exp.query_featurizer(self.exp.nodes[0]))
            batch = self.exp.featurizer(self.exp.nodes[0])
            assert batch.ndim == 1
            plan_feat_size = batch.shape[0]

            # this is by default.
            labels = num_label_bins if p.cross_entropy else 1
            model = treeconv.TreeConvolution(feature_size=query_feat_size,
                                             plan_size=plan_feat_size,
                                             label_size=labels).to(device)
        else:
            # Some training was performed before.  Weights would be re-initialized by initialize_model() below.
            model = self.model

        print('initialize_model curr_value_iter={}'.format(self.curr_value_iter))
        if p.sim:
            should_skip = p.skip_sim_init_iter_1p and hasattr(self, 'model')
            if not should_skip:
                soft_assign_tau = 0.0
                if p.param_tau and hasattr(self, 'model'):
                    # Allows soft assign only if some training has been done.
                    soft_assign_tau = p.param_tau
                if train_from_scratch:
                    print('Training from scratch; forcing tau := 0.')
                    soft_assign_tau = 0.0

                print("Get OrTrainSim in initialize_model")
                self.ema_source_net = self.initialize_model(
                    p=p,
                    model=model,
                    sim=self.get_train_sim_model(),
                    soft_assign_tau=soft_assign_tau,
                    soft_assign_use_ema=p.use_ema_source,
                    ema_source_tm1=self.ema_source_net)
        elif p.param_tau == 0.0:
            print('Reset model to randomized weights!')
            model.reset_weights()

        # Wrap it to get pytorch_lightning niceness.
        model = BalsaModel(
            params=p,
            model=model,
            torch_invert_cost=dataset.TorchInvertCost,
            query_featurizer=self.exp.query_featurizer,
            perturb_query_features=p.perturb_query_features,
            l2_lambda=p.l2_lambda,
            learning_rate=self.lr_schedule.Get()
            if self.adaptive_lr_schedule is None else
            self.adaptive_lr_schedule.Get(),
            optimizer_state_dict=self.prev_optimizer_state_dict,
            reduce_lr_within_val_iter=p.reduce_lr_within_val_iter)
        print('iter', self.curr_value_iter, 'lr', model.learning_rate)
        if p.agent_checkpoint is not None and self.curr_value_iter == 0:
            ckpt = torch.load(p.agent_checkpoint,
                              map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['state_dict'])
            self.model = model.model
            print('Loaded value network checkpoint at iter',
                  self.curr_value_iter)
        if self.curr_value_iter == 0:
            plan_gen_expert_sim.report_model(model)
        return model

    def Run(self):
        p = self.params
        if p.run_baseline:
            print("[Run]. Start to run baseline")
            return self.run_baseline_optimzier()
        else:
            # 2nd run (when we have the baseline run, and worklaod instance saved)
            self.sim = self.get_train_sim_model()
            self.exp, self.exp_val = self._make_exp_buffer()
            # For reporting cleaner hint strings for expert plans, remove their
            # unary ops (e.g., Aggregates).  These calls return copies, so
            # self.{all,train,test}_nodes no longer share any references.
            self.train_nodes = [ele.filter_scans_joins() for ele in self.train_nodes]
            self.test_nodes = [ele.filter_scans_joins() for ele in self.test_nodes]
        print(f"[Run]. Start to run {self.curr_value_iter, p.val_iters}")
        while self.curr_value_iter < p.val_iters:
            begin_iteration = time.time()
            has_timeouts = self.RunOneIter()
            self.LogTimings()
            end_iteration = time.time()
            print(f"[Run]. Train one iteration done, with time={end_iteration - begin_iteration}")
            if (p.early_stop_on_skip_fraction is not None and
                    self.curr_iter_skipped_queries >= p.early_stop_on_skip_fraction * len(self.train_nodes)):
                break

            if p.drop_cache:
                print('Dropping buffer cache.')
                self.db_cli.drop_buffer_cache()

            if p.increment_iter_despite_timeouts:
                # Always increment the iteration counter.  This makes it fairer
                # to compare runs with & without the timeout mechanism (or even between timeout runs).
                self.curr_value_iter += 1
                self.lr_schedule.Step()
                if self.adaptive_lr_schedule is not None:
                    self.adaptive_lr_schedule.Step()
            else:
                if has_timeouts:
                    # Don't count this value iter.
                    # NOTE: it is possible for runs with use_timeout=False to
                    # have timeout events.  This can happen due to pg_executor
                    # encountering an out-of-memory / internal error and
                    # treating an execution as a timeout.
                    pass
                else:
                    self.curr_value_iter += 1
                    self.lr_schedule.Step()
                    if self.adaptive_lr_schedule is not None:
                        self.adaptive_lr_schedule.Step()

        print("[Training] All iteration done")

    def RunOneIter(self):
        p = self.params
        self.curr_iter_skipped_queries = 0
        # Train the model.
        model, dataset = self.train_all_epoch()
        # Replay buffer reset (if enabled).
        if self.curr_value_iter == p.replay_buffer_reset_at_iter:
            self.exp.DropAgentExperience()

        planner = self._make_planner(model, dataset)
        # Use the model to plan the workload.  Execute the plans and get latencies.
        to_execute, execution_results = self.plan_and_execute(model, planner, is_test=False)

        print("[Run]. start FeedbackExecution ----------------------")
        # Add exeuction results to the experience buffer.
        iter_total_latency, has_timeouts = self.FeedbackExecution(
            to_execute, execution_results)
        # Logging.
        if not has_timeouts:
            self.overall_best_train_latency = min(
                self.overall_best_train_latency, iter_total_latency / 1e3)
            to_log = [
                ('latency/workload', iter_total_latency / 1e3,
                 self.curr_value_iter),
                ('latency/workload_best', self.overall_best_train_latency,
                 self.curr_value_iter),
                ('num_query_execs', self.num_query_execs, self.curr_value_iter),
                ('num_queries_with_eps_random', planner.num_queries_with_random,
                 self.curr_value_iter),
                ('curr_iter_skipped_queries', self.curr_iter_skipped_queries,
                 self.curr_value_iter),
                ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
                ('lr', model.learning_rate, self.curr_value_iter),
            ]
            if p.reduce_lr_within_val_iter:
                to_log.append(('iter_final_lr', model.latest_per_iter_lr,
                               self.curr_value_iter))
            self.LogScalars(to_log)

        self.save_running_result(planner, model, iter_total_latency)

        return has_timeouts

    # ---------------- public API ----------------
    def train(self, training_data: List[Dict[str, Any]]):
        pass

    def _train_for_each_epoch(self, train_data_loader: DataLoader):
        pass

    def sql_enhancement(self, query_id: str, sql_str: str, db_cli: PostgresConnector):
        pass

    def search_best_join_order(self, db_cli: PostgresConnector, plan_json_PG, alias, sql_vec, sql,
                               join_list_with_predicate, join_lis):
        pass

    def save(self, epoch=None):
        pass

    def load(self, epoch=None):
        pass

    def _get_data_loader(self, training_data: List[Dict[str, Any]]):
        pass

    def save_running_result(self, planner, model, iter_total_latency):
        print("[Run]. start SaveBestPlans ----------------------")

        self.SaveBestPlans()
        # if (self.curr_value_iter + 1) % 5 == 0:
        #     self.SaveAgent(model, iter_total_latency, curr_value_iter=self.curr_value_iter)

        self.SaveAgent(model, iter_total_latency, curr_value_iter=self.curr_value_iter, parameter=p)

        # Run and log test queries.
        print("[Run]. start EvaluateTestSet ----------------------")
        self.EvaluateTestSet(model, planner)

        # todo: this is fause by default, no need to use here.
        # if p.track_model_moving_averages:
        #     # Update model averages.
        #     # 1. EMA.  Aka Polyak averaging.
        #     if self.curr_value_iter >= 0:
        #         self.UpdateMovingAverage(model,
        #                                  moving_average='ema',
        #                                  ema_decay=p.ema_decay)
        #         if (self.curr_value_iter + 1) % 5 == 0:
        #             # Use EMA to evaluate test set too.
        #             self.SwapMovingAverage(model, moving_average='ema')
        #             # Clear the planner's label cache.
        #             planner.SetModel(model)
        #             self.EvaluateTestSet(model, planner, tag='latency_test_ema')
        #             self.SwapMovingAverage(model, moving_average='ema')
        #
        #     # 2. SWA: Stochastic weight averaging.
        #     if self.curr_value_iter >= 75:
        #         self.UpdateMovingAverage(model, moving_average='swa')
        #         if (self.curr_value_iter + 1) % 5 == 0:
        #             self.SwapMovingAverage(model, moving_average='swa')
        #             # Clear the planner's label cache.
        #             planner.SetModel(model)
        #             self.EvaluateTestSet(model, planner, tag='latency_test_swa')
        #             self.SwapMovingAverage(model, moving_average='swa')

    def train_all_epoch(self, train_from_scratch=False):
        p = self.params
        self.timer.Start('train')

        # --- prepare dataset & loaders ---
        train_ds, train_loader, _, val_loader = self._make_dataloader(log=not train_from_scratch)

        # Use dataset reference (for featurizers / metadata)
        plans_dataset = (
            train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
        )

        # --- build model ---
        model = self._MakeModel(plans_dataset, train_from_scratch)
        if train_from_scratch:
            model.logging_prefix = f"train_from_scratch/iter-{self.curr_value_iter}-"
        else:
            model.logging_prefix = f"train/iter-{self.curr_value_iter}-"

        # --- training loop ---
        print("[Run]. starting train loop...")
        model.train_model(
            train_loader,
            val_loader=val_loader,
            num_epochs=p.epochs,
        )
        print("[Run]. training loop done")

        # --- save trained model reference ---
        self.model = model.model

        # Save optimizer state if requested
        self.prev_optimizer_state_dict = None
        if p.inherit_optimizer_state:
            self.prev_optimizer_state_dict = model.make_optimizer().state_dict()

        self.timer.Stop("train")
        return model, plans_dataset

    def _make_planner(self, model, dataset):
        p = self.params
        if self.sim is not None and self.sim.IsPlanPhysicalButUseGenericOps():
            # With generic Scan/Join removed.
            wi = self.sim._GetPlanner().workload_info
        else:
            wi = self.exp.workload_info
        return BMOptimizer(
            workload_info=wi,
            plan_featurizer=self.exp.featurizer,
            parent_pos_featurizer=self.exp.pos_featurizer,
            query_featurizer=self.exp.query_featurizer,
            # NOTE: unit seems wrong if initialized from SIM.
            inverse_label_transform_fn=dataset.InvertCost,
            model=model,
            plan_physical=p.plan_physical,
            use_plan_restrictions=p.real_use_plan_restrictions)

    def _SampleInternalNode(self, node):
        num_leaves = len(node.leaf_ids())
        num_internal = num_leaves - 1
        assert num_internal > 0, node

        def _Sample(subnode, remaining_internal):
            if len(subnode.children) == 0:
                return None, remaining_internal
            if np.random.rand() < 1. / remaining_internal:
                # Pick this internal node.
                return subnode, None
            # Left branch.
            sampled, rem = _Sample(subnode.children[0], remaining_internal - 1)
            if sampled is not None:
                return sampled, None
            # Right branch.
            return _Sample(subnode.children[1], rem)

        sampled_node, _ = _Sample(node, num_internal)
        return sampled_node

    def SelectPlan(self, found_plans, predicted_latency, found_plan, planner,
                   query_node):
        """Exploration + action selection."""
        p = self.params
        # Sanity check that at most one exploration strategy is specified.
        num_explore_schemes = (p.epsilon_greedy + p.explore_soft_v +
                               p.explore_visit_counts +
                               p.explore_visit_counts_sort +
                               p.explore_visit_counts_latency_sort)
        assert num_explore_schemes <= 1
        if p.epsilon_greedy:
            assert p.epsilon_greedy_random_transform + \
                   p.epsilon_greedy_random_plan <= 1
        if p.epsilon_greedy > 0:
            r = np.random.rand()
            if r < p.epsilon_greedy:
                # Epsilon-greedy policy.
                if p.epsilon_greedy_random_transform:
                    # Randomly transform the best found plan.
                    print('Before: {}'.format(found_plan.hint_str()))
                    sampled_node = self._SampleInternalNode(found_plan)
                    cs = sampled_node.children
                    sampled_node.children = [cs[1], cs[0]]
                    print('After: {}'.format(found_plan.hint_str()))
                elif p.epsilon_greedy_random_plan and self.curr_value_iter > 0:
                    # Randomly pick a plan.
                    predicted_latency, found_plan = planner.SampleRandomPlan(
                        query_node)
                else:
                    # Randomly pick a plan from all found plans.
                    rand_idx = np.random.randint(len(found_plans))
                    predicted_latency, found_plan = found_plans[rand_idx]
        elif p.explore_soft_v:
            # Sample proportional to exp (-V_theta(s)).
            with torch.no_grad():
                v_values = torch.tensor([-v for v, _ in found_plans],
                                        dtype=torch.float32)
                v_values -= v_values.max()
                softmax = torch.softmax(v_values, dim=0)
                rand_idx = torch.multinomial(softmax, num_samples=1).item()
            predicted_latency, found_plan = found_plans[rand_idx]
        elif (p.explore_visit_counts or p.explore_visit_counts_sort or
              p.explore_visit_counts_latency_sort):
            visit_counts = np.zeros(len(found_plans), dtype=np.float32)
            query_name = query_node.info['query_name']
            for i, (_, plan) in enumerate(found_plans):
                hint_str = plan.hint_str(with_physical_hints=p.plan_physical)
                visit_counts[i] = self.query_execution_cache.GetVisitCount(key=(query_name, hint_str))
            visit_sum = visit_counts.sum()
            if visit_sum > 0:
                # If none are visited, skip this step.
                if p.explore_visit_counts:
                    # Sample proportional to
                    #    visit_sum / (1 + num_visits(plan_i))
                    # Disregarding predicted V() is sort of saying they are
                    # probably all similar, let's just use visit counts.
                    scores = visit_sum * 1.0 / (1.0 + visit_counts)
                    with torch.no_grad():
                        scores = torch.from_numpy(scores)
                        rand_idx = torch.multinomial(scores,
                                                     num_samples=1).item()
                    predicted_latency, found_plan = found_plans[rand_idx]
                    print('counts', visit_counts, 'sampled_idx', rand_idx,
                          'sampled_cnt', visit_counts[rand_idx])
                else:
                    # Sort by (visit count, predicted latency).  Execute the
                    # smallest.
                    assert p.explore_visit_counts_sort or \
                           p.explore_visit_counts_latency_sort
                    # Cast to int so the debug messages look nicer.
                    visit_counts = visit_counts.astype(np.int32, copy=False)

                    # If all plans have been visited, sort by predicted latency.
                    if (p.explore_visit_counts_latency_sort and
                            all([x > 0 for x in visit_counts])):
                        found_plans_sorted, visit_counts_sorted = zip(
                            *sorted(zip(found_plans, visit_counts),
                                    key=lambda tup: tup[0][0]))
                    else:
                        found_plans_sorted, visit_counts_sorted = zip(
                            *sorted(zip(found_plans, visit_counts),
                                    key=lambda tup: (tup[1], tup[0][0])))
                    predicted_latency, found_plan = found_plans_sorted[0]
                    print(
                        'selected cnt,latency=({}, {});'.format(
                            visit_counts_sorted[0], predicted_latency),
                        'sorted:',
                        list(
                            zip(visit_counts_sorted,
                                map(lambda tup: tup[0], found_plans_sorted))))

        return predicted_latency, found_plan

    def plan_and_execute(self, model, planner: BMOptimizer, is_test=False):
        p = self.params
        model.eval()
        to_execute = []
        execution_results = []
        positions_of_min_predicted = []

        if p.sim:
            print("Get OrTrainSim in plan_and_execute")
            sim = self.get_train_sim_model()

        nodes = self.test_nodes if is_test else self.train_nodes
        if not is_test:
            self.timeout_controller.OnIterStart()

        # planner_config = None
        # if p.planner_config is not None:
        #     planner_config = search_opt.PlannerConfig.Get(p.planner_config)

        epsilon_greedy_within_beam_search = 0
        if not is_test and p.epsilon_greedy_within_beam_search > 0:
            epsilon_greedy_within_beam_search = p.epsilon_greedy_within_beam_search

        self.timer.Start('plan_test_set' if is_test else 'plan')

        query_execution_statistics = dict()

        for i, node in enumerate(nodes):
            print('---------------------------------------')
            inference_start = time.time()
            planning_time, found_plan, predicted_latency, found_plans = planner.plan(
                query_node=node,
                return_all_found=True,
                verbose=False,
                avoid_eq_filters=is_test and p.avoid_eq_filters,
                epsilon_greedy=epsilon_greedy_within_beam_search,
            )

            predicted_latency, found_plan = self.SelectPlan(
                found_plans, predicted_latency, found_plan, planner, node
            )
            print('{}q{}, predicted time: {:.1f}'.format(
                '[Test set] ' if is_test else '', node.info['query_name'], predicted_latency
            ))

            predicted_costs = None
            if p.sim:
                predicted_costs = sim.predict(node, [fp for _, fp in found_plans])

            node.info['curr_predicted_latency'] = planner.infer(node, [node])[0]
            self.LogScalars([(
                f'predicted_latency_expert_plans/q{node.info["query_name"]}',
                node.info['curr_predicted_latency'] / 1e3,
                self.curr_value_iter
            )])

            hint_str = found_plan.hint_str(with_physical_hints=p.plan_physical)
            query_inference_time = time.time() - inference_start

            if is_test:
                curr_timeout = 180000
            else:
                curr_timeout = self.timeout_controller.GetTimeout(node)

            print(f'q{node.info["query_name"]},(predicted {predicted_latency:.1f}),{hint_str}')

            to_execute.append((
                node.info['sql_str'], hint_str, planning_time,
                found_plan, predicted_latency, curr_timeout
            ))

            # Statistic for logging inference + planning + execution time lateron
            q_exec_stat = {
                'query_name': node.info['query_name'],
                'sql_str': node.info['sql_str'],
                'hint_str': hint_str,
                'inference_time': query_inference_time,
                'predicted_latency': predicted_latency  # This is the predicted latency in milliseconds
            }

            query_execution_statistics[q_exec_stat['query_name']] = q_exec_stat

            # --- Direct execution ---
            sql_with_hint = plan_gen_expert_sim._fuse_hints(node.info['sql_str'], hint_str)
            try:
                plan_json, _ = self.db_cli.explain_analysis_read_sql(sql_with_hint)
                real_exe_time = plan_json['Execution Time']

            except Exception as e:
                print(f"Execution failed for {node.info['query_name']} with {e}")
                plan_json = None
                real_exe_time = -1

            log_str = self.generate_query_search_history(
                real_cost=real_exe_time,
                query_name=node.info['query_name'],
                hint_str=hint_str,
                hinted_plan=found_plan,
                query_node=node,
                predicted_latency=predicted_latency,
                curr_timeout_ms=curr_timeout,
                found_plans=found_plans,
                predicted_costs=predicted_costs,
                is_test=is_test
            )

            try:
                self.check_hint_respected(
                    plan_json=plan_json,
                    query_name=node.info['query_name'],
                    sql_str=node.info['sql_str'],
                    hint_str=hint_str,
                    plan_physical=p.plan_physical)
            except Exception as e:
                print(f"[error] Exception in ParseExecutionResult {e}")

            # Update stats
            if plan_json is None:
                q_exec_stat['execution_time'] = -1
                q_exec_stat['planning_time'] = -1
            else:
                q_exec_stat['execution_time'] = plan_json['Execution Time']
                q_exec_stat['planning_time'] = plan_json['Planning Time']

            print(
                f"\t{q_exec_stat['query_name']}: "
                f"Inference {q_exec_stat['inference_time']:.4f} "
                f"Planning {q_exec_stat['planning_time']:.4f} "
                f"Execution {q_exec_stat['execution_time']:.4f}"
            )

            execution_results.append(log_str)

            # Track the cheapest predicted plan position
            min_p_latency, min_pos = min((pl, pos) for pos, (pl, _) in enumerate(found_plans))
            positions_of_min_predicted.append(min_pos)

        self.timer.Stop('plan_test_set' if is_test else 'plan')

        # Logging to file (kept same as your old code)
        # if 'cls' in wandb.run.config:
        #     experiment_cls = wandb.run.config['cls'].split('/')[-1]
        # else:
        #     experiment_cls = 'unknown'
        experiment_cls = 'unknown'

        query_log_file_name = f"logs/{self.time_temp}__{experiment_cls}__plan_and_execute.txt"

        with open(query_log_file_name, 'a') as qlf:
            for k in query_execution_statistics.keys():
                curr = query_execution_statistics[k]
                mse = ((curr['execution_time'] - curr['predicted_latency']) / 1e3) ** 2
                output_string = (
                    f"{curr['query_name']};{curr['inference_time']:.4f};"
                    f"{curr['planning_time']:.4f};{curr['execution_time']:.4f};"
                    f"{curr['predicted_latency']:.4f};{mse:.4f}"
                )
                qlf.write(output_string + os.linesep)

        if is_test:
            try:
                query_log_file_name_json = f"logs/{self.time_temp}__{experiment_cls}__plan_and_execute_running_stastics.jsonl"
                with open(query_log_file_name_json, "a") as f:
                    f.write(json.dumps(query_execution_statistics) + "\n")
            except Exception as e:
                print(f" save plans logs has error {e}")

        return to_execute, execution_results

    def FeedbackExecution(self, to_execute, execution_results):
        p = self.params
        results = []
        iter_total_latency = 0
        iter_max_latency = 0
        has_timeouts = False
        num_timeouts = 0
        # Errors the current policy incurs on (agent plans for train queries,
        # expert plans for train queries).
        agent_plans_diffs = []
        expert_plans_diffs = []
        for node, result_tup, to_execute_tup in zip(self.train_nodes,
                                                    execution_results,
                                                    to_execute):
            try:
                result, real_cost, server_ip = result_tup
                _, hint_str, planning_time, actual, predicted_latency, \
                curr_timeout = to_execute_tup
                # Record execution result, potentially with real_cost = -1
                # indicating a timeout.  The cache would only record a lower
                # latency value so once it gets a -1 label for a plan, it'd not be
                # updated again.  If a future iteration this plan is still
                # selected, it'd get the same -1 label from the cache, ensuring
                # that has_timeouts below would be set to True correctly.
                self.query_execution_cache.Put(key=(node.info['query_name'],
                                                    hint_str),
                                               value=result_tup,
                                               latency=real_cost)
                self.timeout_controller.RecordQueryExecution(node, real_cost)

                # Process timeout.
                # FIXME: even when use_timeout=False, pg_executor may treat a rare
                # InternalError_ or OperationalError as a timeout event.  These are
                # rare but could incorrectly get a timeout label below.  We should
                # fix this by marking a Node as a timeout & allowing Experience to
                # skip featurizing those marked nodes.
                if real_cost < 0:
                    has_timeouts = True
                    num_timeouts += 1
                    self.num_total_timeouts += 1
                    if p.special_timeout_label:
                        real_cost = self.timeout_label()
                        print('Timeout detected! Assigning a special label',
                              real_cost, '(server_ip={})'.format(server_ip))
                    else:
                        real_cost = curr_timeout * 2
                        print('Timeout detected! Assigning 2*timeout as label',
                              real_cost, '(server_ip={})'.format(server_ip))
                    # At this point, 'actual' is a Node produced from the agent
                    # consisting of just scan/join nodes.  It has gone through hint
                    # checks in ParseExecutionResult() -- i.e., it should be the
                    # same as the EXPLAIN result from a local PG with an
                    # agent-produced hint string.
                    #
                    # We manually fill in this field for hindsight labeling (if
                    # enabled) to work.  Intermediate goals are not collected since
                    # we don't know what those "sub-latencies" are.
                    actual.actual_time_ms = real_cost
                    # Mark a special timeout field.
                    actual.is_timeout = True
                else:
                    agent_plans_diffs.append((real_cost - predicted_latency) / 1e3)
                expert_plans_diffs.append(
                    (node.cost - node.info['curr_predicted_latency']) / 1e3)

                assert real_cost > 0, real_cost
                actual.cost = real_cost
                actual.info = copy.deepcopy(node.info)
                actual.info.pop('explain_json')

                # Put into experience/replay buffer.
                self.exp.add(actual)
                # Update the best plan cache.
                self.best_plans.Put(key=node.info['query_name'],
                                    value=actual,
                                    latency=real_cost)

                # Logging.
                results.append(result)
                iter_total_latency += real_cost
                iter_max_latency = max(iter_max_latency, real_cost)
                self.LogScalars([
                    ('latency/q{}'.format(node.info['query_name']), real_cost / 1e3,
                     self.curr_value_iter),
                    # Max per-query latency in this iter.  This bounds
                    # the time required for query execution if we were
                    # to parallelize everything.
                    ('curr_iter_max_ms', iter_max_latency, self.curr_value_iter),
                ])
            except Exception as e:
                print(f"Wrong during the FeedbackExecution collection {e}")

        # Logging.
        self.LogScalars([
            # Prediction errors.
            ('latency/mean_l1_agent_secs', np.mean(np.abs(agent_plans_diffs)),
             self.curr_value_iter),
            ('latency/mean_pred-tgt_agent_secs', -np.mean(agent_plans_diffs),
             self.curr_value_iter),
            ('latency/mean_l1_expert_secs', np.mean(np.abs(expert_plans_diffs)),
             self.curr_value_iter),
            ('latency/mean_pred-tgt_expert_secs', -np.mean(expert_plans_diffs),
             self.curr_value_iter),
            # Timeout metrics.
            ('curr_timeout',
             curr_timeout / 1e3 if curr_timeout is not None else 0,
             self.curr_value_iter),
            ('num_total_timeouts', self.num_total_timeouts,
             self.curr_value_iter),
            ('num_timeouts', num_timeouts, self.curr_value_iter),
            # X-axis.
            ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
        ])

        return iter_total_latency, has_timeouts

    def _make_dataloader(self, log=True):
        p = self.params
        skip_first_n = 0
        on_policy = p.on_policy
        data = self.exp.featurize(
            rewrite_generic=not p.plan_physical,
            verbose=False,
            skip_first_n=skip_first_n,
            deduplicate=p.dedup_training_data,
            physical_execution_hindsight=p.physical_execution_hindsight,
            on_policy=on_policy,
            use_last_n_iters=p.use_last_n_iters,
            use_new_data_only=p.use_new_data_only,
            skip_training_on_timeouts=p.skip_training_on_timeouts)
        # [np.ndarray], torch.Tensor, torch.Tensor, [float].
        all_query_vecs, all_feat_vecs, all_pos_vecs, all_costs = data[:4]
        train_ds, train_loader, val_ds, val_loader, num_train, num_validation = expert_datasets.graphexp_make_dataloader(
            data=data,
            label_transforms=p.label_transforms,
            validate_fraction=p.validate_fraction,
            batch_size=p.bs
        )
        print('num_train={} num_validation={}'.format(num_train, num_validation))

        if log:
            train_labels = np.asarray(all_costs)[train_ds.indices]
            self._LogDatasetStats(train_labels, None)
        return train_ds, train_loader, val_ds, val_loader

    @staticmethod
    def initialize_model(p,
                         model,
                         sim,
                         soft_assign_tau=0.0,
                         soft_assign_use_ema=False,
                         ema_source_tm1=None):
        """Initializes model weights.

        Given model_(t-1), sim, ..., ema_source_tm1, initializes model_t as follows.

        If soft_assign_use_ema is False:

            model := soft_assign_tau*model + (1-soft_assign_tau)*sim.

            In particular:
            - soft_assign_tau = 0 means always reinitializes 'model' with 'sim'.
            - soft_assign_tau = 1 means don't reinitialize 'model'; keep training it
                across value iterations.

            A value of 0.1 seems to perform well.

        Otherwise, use an exponential moving average of "source networks":

            source_t = soft_assign_tau * source_(t-1)
                         + (1-soft_assign_tau) model_(t-1)
            model_t := source_t

            In particular:
            - soft_assign_tau = 0 means don't reinitialize 'model'; keep training it
                across value iterations.
            - soft_assign_tau = 1 means always reinitializes 'model' with 'sim'.

            A value of 0.05 seems to perform well.

        For both schemes, before training 'model' for the very first time it is
        always initialized with the simulation model 'sim'.

        Args:
          p: params.
          model: current iteration's value model.
          sim: the trained-in-sim model.
          soft_assign_tau: if positive, soft initializes 'model' using the formula
            described above.
          soft_assign_use_ema: whether to use an exponential moving average of
            "source networks".
          ema_source_tm1: the EMA of source networks at iteration t-1.
        """

        def Rename(state_dict):
            new_state_dict = collections.OrderedDict()
            for key, value in state_dict.items():
                new_key = key
                if key.startswith('tree_conv.'):
                    new_key = key.replace('tree_conv.', '')
                new_state_dict[new_key] = value
            return new_state_dict

        sim_weights = sim.model.state_dict()
        sim_weights_renamed = copy.deepcopy(Rename(sim_weights))
        model_weights = model.state_dict()
        assert model_weights.keys() == sim_weights_renamed.keys()

        tau = soft_assign_tau
        if tau:
            if not soft_assign_use_ema:
                print('Assigning real model := {}*SIM + {}*previous real model'.
                      format(1 - tau, tau))
                for key, param in model_weights.items():
                    param.requires_grad = False
                    param = param * tau + sim_weights_renamed[key] * (1.0 - tau)
                    param.requires_grad = True
            else:
                # Use an exponential moving average of source networks.
                if ema_source_tm1 is None:
                    ema_source_tm1 = sim_weights_renamed
                assert isinstance(ema_source_tm1,
                                  collections.OrderedDict), ema_source_tm1
                assert ema_source_tm1.keys() == model_weights.keys()
                # Calculates source_t for current iteration t:
                #    source_t = tau * source_(t-1) + (1-tau) model_(t-1)
                with torch.no_grad():
                    ema_source_t = copy.deepcopy(ema_source_tm1)
                    for key, param in model_weights.items():
                        ema_source_t[key] = tau * ema_source_tm1[key] + (
                                1.0 - tau) * param
                # Assign model_t := source_t.
                model.load_state_dict(ema_source_t)
                print('Initialized from EMA source network: tau={}'.format(tau))
                # Return source_t for next iter's use.
                return ema_source_t
        else:
            model.load_state_dict(sim_weights_renamed)
            print('Initialized from SIM weights.')

        if p.finetune_out_mlp_only:
            for name, param in model.named_parameters():
                if 'out_mlp' not in name:
                    param.detach_()
                    param.requires_grad = False
                    print('Freezing', name)

        if p.param_noise:
            for layer in model.out_mlp:
                if isinstance(layer, nn.Linear):
                    print('Adding N(0, {}) to out_mlp\'s {}.'.format(
                        p.param_noise, layer))

                    def _Add(w):
                        w.requires_grad = False
                        w.add_(
                            torch.normal(mean=0.0,
                                         std=p.param_noise,
                                         size=w.shape,
                                         device=w.device))
                        w.requires_grad = True

                    _Add(layer.weight)

    # ---------------- logs ----------------

    def timeout_label(self):
        return 4096 * 1000

    def LogTimings(self):
        """Logs timing statistics."""
        p = self.params
        stages = ['train', 'plan', 'wait_for_executions']
        num_iters_done = self.curr_value_iter + 1
        if p.test_query_glob is not None and \
                num_iters_done >= p.test_after_n_iters and \
                num_iters_done % p.test_every_n_iters == 0:
            stages += ['plan_test_set', 'wait_for_executions_test_set']
        timings = [self.timer.GetLatestTiming(s) for s in stages]
        iter_total_s = sum(timings)
        cumulative_timings = [self.timer.GetTotalTiming(s) for s in stages]
        total_s = sum(cumulative_timings)
        data_to_log = []
        for stage, timing, cumulative_timing in zip(stages, timings,
                                                    cumulative_timings):
            data_to_log.extend([
                # Time, this iter.
                ('timing/{}'.format(stage), timing, self.curr_value_iter),
                # %Time of this iter's total.
                ('timing_pct/{}'.format(stage), timing / iter_total_s,
                 self.curr_value_iter),
                # Total time since beginning.
                ('timing_cumulative/{}'.format(stage), cumulative_timing,
                 self.curr_value_iter),
                # Total %time of all iters so far.
                ('timing_cumulative_pct/{}'.format(stage),
                 cumulative_timing / total_s, self.curr_value_iter),
            ])
        # X-axis.
        data_to_log.append(
            ('curr_value_iter', self.curr_value_iter, self.curr_value_iter))
        self.LogScalars(data_to_log)

    def _LogDatasetStats(self, train_labels, num_new_datapoints):
        # Track # of training trees that are not timeouts.
        num_normal_trees = (np.asarray(train_labels) !=
                            self.timeout_label()).sum()
        data = [
            ('train/iter-{}-num-trees'.format(self.curr_value_iter),
             len(train_labels), self.curr_value_iter),
            ('train/num-trees', len(train_labels), self.curr_value_iter),
            ('train/iter-{}-num-normal-trees'.format(self.curr_value_iter),
             num_normal_trees, self.curr_value_iter),
            ('train/num-normal-trees', num_normal_trees, self.curr_value_iter),
            ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
        ]
        if num_new_datapoints is not None:
            data.append(('train/num-new-datapoints', num_new_datapoints,
                         self.curr_value_iter))
        self.LogScalars(data)

    def LogExpertExperience(self, expert_train_nodes: List[workload.Node], expert_test_nodes: List[workload.Node]):
        p = self.params
        total_s = 0
        data_to_log = []
        num_joins = []
        for node in expert_train_nodes:
            # Real latency in ms was assigned to node.cost as impl convenience.
            data_to_log.append(
                ('latency_expert/q{}'.format(node.info['query_name']),
                 node.cost / 1e3, 0))
            total_s += node.cost / 1e3
            num_joins.append(len(node.leaf_ids()) - 1)
        data_to_log.append(('latency_expert/workload', total_s, 0))
        print('latency_expert/workload (seconds): {:.2f} ({} queries)'.format(total_s, len(expert_train_nodes)))

        if p.test_query_glob is not None:
            total_s_test = 0
            for node in expert_test_nodes:
                data_to_log.append(('latency_expert_test/q{}'.format(node.info['query_name']), node.cost / 1e3, 0))
                total_s_test += node.cost / 1e3
                num_joins.append(len(node.leaf_ids()) - 1)
            data_to_log.append(('latency_expert_test/workload', total_s_test, 0))
            print('latency_expert_test/workload (seconds): {:.2f} ({} queries)'.
                  format(total_s_test, len(expert_test_nodes)))
        data_to_log.append(('curr_value_iter', 0, 0))
        self.LogScalars(data_to_log)
        print('Number of joins [{}, {}], avg {:.1f}'.format(np.min(num_joins), np.max(num_joins), np.mean(num_joins)))

    def LogScalars(self, metrics):
        if not isinstance(metrics, list):
            assert len(metrics) == 3, 'Expected (tag, val, global_step)'
            metrics = [metrics]
        for tag, val, global_step in metrics:
            self.summary_writer.add_scalar(tag, val, global_step=global_step)
        d = dict([(tag, val) for tag, val, _ in metrics])
        assert len(set([gs for _, _, gs in metrics])) == 1, metrics
        # self.wandb_logger.log_metrics(d)

    def SaveBestPlans(self):
        """Saves the best plans found so far.

        Write to best_plans/, under the run directory managed by wandb:

        - <query_name>.sql: best plan for each query as a commented .sql file
        - all.sql: all commented sql texts concatenated together
        - latencies.txt: a CSV containing "query name -> latency", including an
            "all" entry for the total latency
        - plans.pkl: a map {query name -> plans_lib.Node, the best plan}

        The all.sql file (or the individual query sql files) are ready to be
        piped to a SQL shell for execution.
        """
        p = self.params
        best_plans_dir = os.path.join(self.wandb_logger.experiment.dir,
                                      'best_plans/')
        qnames = []
        latencies = []
        sqls = []
        total_ms = 0
        all_nodes = {}
        # For calling wandb.save() to continuously upload on update.
        # w = self.wandb_logger.experiment
        # wandb_dir = w.dir
        # Save all plans.
        for query_name, value_tup in sorted(self.best_plans._cache.items()):
            best_node, best_latency = value_tup
            all_nodes[query_name] = best_node
            hint = best_node.hint_str(with_physical_hints=p.plan_physical)
            # sql = PostgresAddCommentToSql(best_node.info['sql_str'], hint)
            sql = hint + '\n' + best_node.info['sql_str']

            path = io.save_text(
                sql, os.path.join(best_plans_dir, '{}.sql'.format(query_name)))
            # w.save(path, base_path=wandb_dir)
            sqls.append(sql)
            qnames.append(query_name)
            latencies.append(best_latency)
            total_ms += best_latency
        qnames.append('all')
        latencies.append(total_ms)
        # all.sql.
        # path = io.save_text('\n'.join(sqls), os.path.join(best_plans_dir,  'all.sql'))
        # w.save(path, base_path=wandb_dir)
        # latencies.txt.
        # pd.DataFrame({
        #     'query': qnames,
        #     'latency_ms': latencies
        # }).to_csv(os.path.join(best_plans_dir, 'latencies.txt'),
        #           header=True,
        #           index=False)
        # w.save(os.path.join(best_plans_dir, 'latencies.txt'),
        #        base_path=wandb_dir)
        # plans.pkl.
        # path = Save(all_nodes, os.path.join(best_plans_dir, 'plans.pkl'))
        # w.save(path, base_path=wandb_dir)

    def EvaluateTestSet(self, model, planner, tag='latency_test'):
        # TODO: exclude running time for evaluating test set.
        p = self.params
        num_iters_done = self.curr_value_iter + 1
        if p.test_query_glob is None or \
                num_iters_done < p.test_after_n_iters or \
                num_iters_done % p.test_every_n_iters != 0:
            return
        if p.test_using_retrained_model:
            print(
                '[Test set] training a new model just for test set reporting.')
            # Retrain a new 'model' and build a 'planner'.
            model, dataset = self.train_all_epoch(train_from_scratch=True)
            planner = self._make_planner(model, dataset)
        to_execute_test, execution_results_test = self.plan_and_execute(
            model, planner, is_test=True)
        self.LogTestExperience(to_execute_test, execution_results_test, tag=tag)

    def SaveAgent(self, model, iter_total_latency, curr_value_iter=None, parameter=None):
        """Saves the complete execution state of the agent."""
        # TODO: not complete state, currently missing:
        #  - query exec cache
        #  - moving averages
        #  - a bunch of fields (see Run())
        # TODO: support reloading & resume.

        # Determine save directory: use parameter.model_save_path if available, else Wandb dir
        base_save_dir = parameter.model_save_path
        ckpt_path = os.path.join(base_save_dir, f'{parameter.model_prefix}_checkpoint.pt')
        print(f"\n------ saving result into {ckpt_path} ------\n")

        # Manually construct state dictionary to avoid distributed hooks
        try:
            state_dict = {}
            for name, param in model.named_parameters():
                state_dict[name] = param.detach().cpu()
            for name, buffer in model.named_buffers():
                state_dict[name] = buffer.detach().cpu()
            os.makedirs(base_save_dir, exist_ok=True)
            torch.save(state_dict, ckpt_path)
        except Exception as e:
            print(f"Error saving checkpoint to {ckpt_path}: {e}")
            raise

        # Saving intermediate checkpoints
        if curr_value_iter is not None:
            base_folder_path = os.path.join(base_save_dir, f'{parameter.model_prefix}_checkpoints', self.time_temp)
            if not os.path.exists(base_folder_path):
                os.makedirs(base_folder_path)
            intermediate_path = os.path.join(base_folder_path, f'checkpoint__iter{curr_value_iter}.pt')
            try:
                torch.save(state_dict, intermediate_path)
            except Exception as e:
                print(f"Error saving intermediate checkpoint to {intermediate_path}: {e}")
                raise

        # Save metadata
        io.save_text(
            'value_iter,{}'.format(self.curr_value_iter),
            os.path.join(base_save_dir, f'{parameter.model_prefix}_checkpoint-metadata.txt'))
        print('Saved iter={} checkpoint to: {}'.format(self.curr_value_iter, ckpt_path))

        # Replay buffer. Saved under data/.
        self._SaveReplayBuffer(iter_total_latency)

    def Save(self, path):
        """Saves all Nodes in the current replay buffer to a file."""
        if os.path.exists(path):
            old_path = path
            path = '{}-{}'.format(old_path, time.time())
            print('Path {} exists, appending current time: {}'.format(
                old_path, path))
            assert not os.path.exists(path), path
        to_save = (self.initial_size, self.nodes)
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)
        print('Saved Experience to:', path)

    def LogTestExperience(self,
                          to_execute_test,
                          execution_results,
                          tag='latency_test'):
        assert len(self.test_nodes) == len(execution_results)
        iter_total_latency = 0
        rows = []
        data = []
        has_timeouts = False
        # Errors the current policy incurs on (agent plans for test queries,
        # expert plans for test queries).
        agent_plans_diffs = []
        expert_plans_diffs = []
        for node, to_execute, result_tup in zip(self.test_nodes,
                                                to_execute_test,
                                                execution_results):
            _, real_cost, _ = result_tup
            if real_cost < 0:
                has_timeouts = True
                break
            iter_total_latency += real_cost
            rows.append((node.info['query_name'], real_cost / 1e3,
                         self.curr_value_iter))
            data.append(('{}/q{}'.format(tag, node.info['query_name']),
                         real_cost / 1e3, self.curr_value_iter))
            # Tracks prediction errors.
            agent_plans_diffs.append((real_cost - to_execute[-2]) / 1e3)
            expert_plans_diffs.append(
                (node.cost - node.info['curr_predicted_latency']) / 1e3)
        if has_timeouts:
            # "Timeouts" for test set queries are rare events such as
            # out-of-disk errors due to a lot of intermediate results being
            # written out.
            print(
                '[Test set {}] timeout events detected during eval'.format(tag))
            return
        # Log a table of latencies, sorted by descending latency.
        rows = list(sorted(rows, key=lambda r: r[1], reverse=True))
        # table = wandb.Table(columns=['query_name', tag, 'curr_value_iter'],
        #                     rows=rows)
        # self.wandb_logger.experiment.log({'{}_table'.format(tag): table})

        data.extend([
            (tag + '/workload', iter_total_latency / 1e3, self.curr_value_iter),
            (tag + '/mean_l1_agent_secs', np.mean(np.abs(agent_plans_diffs)),
             self.curr_value_iter),
            (tag + '/mean_pred-tgt_agent_secs', -np.mean(agent_plans_diffs),
             self.curr_value_iter),
            (tag + '/mean_l1_expert_secs', np.mean(np.abs(expert_plans_diffs)),
             self.curr_value_iter),
            (tag + '/mean_pred-tgt_expert_secs', -np.mean(expert_plans_diffs),
             self.curr_value_iter),
            ('num_query_execs', self.num_query_execs, self.curr_value_iter),
            ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
        ])
        if tag == 'latency_test':
            self.overall_best_test_latency = min(self.overall_best_test_latency,
                                                 iter_total_latency / 1e3)
            val_to_log = self.overall_best_test_latency
        elif tag == 'latency_test_swa':
            self.overall_best_test_swa_latency = min(
                self.overall_best_test_swa_latency, iter_total_latency / 1e3)
            val_to_log = self.overall_best_test_swa_latency
        else:
            assert tag == 'latency_test_ema', tag
            self.overall_best_test_ema_latency = min(
                self.overall_best_test_ema_latency, iter_total_latency / 1e3)
            val_to_log = self.overall_best_test_ema_latency
        data.append((tag + '/workload_best', val_to_log, self.curr_value_iter))
        self.LogScalars(data)

    def _SaveReplayBuffer(self, iter_total_latency):
        p = self.params
        # "<class 'experiments.ConfigName'>" -> "ConfigName".
        experiment = str(p.cls).split('.')[-1][:-2]
        path = 'data/replay-{}-{}execs-{}nodes-{}s-{}iters-{}.pkl'.format(
            experiment, self.num_query_execs, len(self.exp.nodes),
            int(iter_total_latency / 1e3), self.curr_value_iter,
            "self.wandb_logger.experiment.id"
        )
        self.exp.Save(path)
        # Remove previous.
        if self._latest_replay_buffer_path is not None:
            os.remove(self._latest_replay_buffer_path)
        self._latest_replay_buffer_path = path

    @staticmethod
    def generate_query_search_history(
            real_cost: float,
            query_name: str,
            hint_str: str,
            hinted_plan: workload.Node,
            query_node: workload.Node,
            predicted_latency: float,
            curr_timeout_ms: Optional[int] = None,
            found_plans: Optional[List[Tuple[float, workload.Node]]] = None,
            predicted_costs: Optional[List[Optional[float]]] = None,
            is_test: bool = False) -> str:
        messages = []
        messages.append('{}Running {}: hinted plan\n{}'.format(
            '[Test set] ' if is_test else '', query_name, hinted_plan))
        messages.append('filters')
        messages.append(pprint.pformat(query_node.info['all_filters']))
        messages.append('')
        messages.append('q{},{:.1f},{}'.format(query_node.info['query_name'],
                                               real_cost, hint_str))
        messages.append(
            '{} Execution time: {:.1f} (predicted {:.1f}) curr_timeout_ms={}'.
                format(query_name, real_cost, predicted_latency, curr_timeout_ms))
        messages.append('Expert plan: latency, predicted, hint')
        expert_hint_str = query_node.hint_str()
        expert_hint_str_physical = query_node.hint_str(with_physical_hints=True)
        messages.append('  {:.1f} (predicted {:.1f})  {}'.format(
            query_node.cost, query_node.info['curr_predicted_latency'],
            expert_hint_str))
        if found_plans:
            if predicted_costs is None:
                predicted_costs = [None] * len(found_plans)
            messages.append('SIM-predicted costs, predicted latency, plan: ')
            min_p_latency = np.min([p_latency for p_latency, _ in found_plans])
            for p_cost, found in zip(predicted_costs, found_plans):
                p_latency, found_plan = found
                found_hint_str = found_plan.hint_str()
                found_hint_str_physical = found_plan.hint_str(with_physical_hints=True)
                extras = [
                    'cheapest' if p_latency == min_p_latency else '',
                    '[expert plan]'
                    if found_hint_str_physical == expert_hint_str_physical else '',
                    '[picked]' if found_hint_str_physical == hint_str else ''
                ]
                extras = ' '.join(filter(lambda s: s, extras)).strip()
                if extras:
                    extras = '<-- {}'.format(extras)
                if p_cost:
                    messages.append('  {:.1f}  {:.1f}  {}  {}'.format(p_cost, p_latency, found_hint_str, extras))
                else:
                    messages.append('          {:.1f}  {}  {}'.format(p_latency, found_hint_str, extras))
        messages.append('-' * 80)
        return '\n'.join(messages)

    def check_hint_respected(
            self,
            plan_json: Dict,
            query_name: str,
            sql_str: str,
            hint_str: str,
            plan_physical: bool = True):
        if hint_str is not None:
            # Check that the hint has been respected.  No need to check if running baseline.
            #
            # lehl@2024-07-04: Because we included bitmap and tid scans into the allowed scan ops,
            # the back-parsed executed hint str will not match the sent one, as for example
            # hash joins are replaced with nested loop joins if there is a bitmap scan underneath.
            #
            do_hint_check = False

            # default to use here, if not timeout,
            if plan_json is not None:
                executed_node = workload.Node.plan_json_to_node(plan_json)
            else:
                # Timeout has occurred & 'result' is empty.  Fallback to checking against local Postgres.
                print('Timeout occurred; checking the hint against local PG.')
                executed_node, _ = plan_node.PGToNodeHelper.sql_to_plan_node(cursor=self.db_cli, sql=sql_str,
                                                                             comment=hint_str)
            executed_node = executed_node.filter_scans_joins()
            executed_hint_str = executed_node.hint_str(with_physical_hints=plan_physical)

            if do_hint_check and hint_str != executed_hint_str:
                print('initial\n', hint_str)
                print('after\n', executed_hint_str)
                msg = 'Hint not respected for {}'.format(query_name)
                try:
                    assert False, msg
                except Exception as e:
                    print(e, flush=True)
                    import ipdb
                    ipdb.set_trace()


if __name__ == '__main__':
    import argparse
    from data_collection.collect_data_unified import BufferManager
    from data_collection.collect_data_unified import read_sql_files
    from common import hyperparams


    class BalsaParams:
        """Params for run.BalsaAgent."""

        @classmethod
        def Params(cls):
            p = hyperparams.InstantiableParams(cls)
            p.define('db', 'imdbload', 'Name of the Postgres database.')

            p.define('query_dir', 'queries/join-order-benchmark',
                     'Directory of the .sql queries.')
            p.define(
                'query_glob', '*.sql',
                'If supplied, glob for this pattern. Otherwise, use all queries. Example: 29*.sql.'
            )
            p.define(
                'test_query_glob', None,
                'Similar usage as query_glob. If None, treat all queries as training nodes.'
            )

            p.define('engine', 'postgres',
                     'The execution engine.  Options: postgres.')
            p.define('engine_dialect_query_dir', None,
                     'Directory of the .sql queries in target engine\'s dialect.')

            p.define('run_baseline', False,
                     'If true, just load the queries and run them.')

            p.define(
                'drop_cache', True,
                'If true, drop the buffer cache at the end of each value iteration.'
            )
            p.define(
                'plan_physical', True,
                'If true, plans physical scan/join operators.  ' \
                'Otherwise, just join ordering.'
            )

            p.define('cost_model', 'postgrescost',
                     'A choice of postgrescost, mincardcost.')

            p.define('bushy', True, 'Plans bushy query execution plans.')

            p.define('search_space_join_ops',
                     ['Hash Join', 'Merge Join', 'Nested Loop'],
                     'Action space: join operators to learn and use.')
            p.define('search_space_scan_ops',
                     ['Index Scan', 'Index Only Scan', 'Seq Scan', 'Bitmap Heap Scan', 'Tid Scan'],
                     'Action space: scan operators to learn and use.')

            # LR.
            p.define('lr', 1e-3, 'Learning rate.')
            p.define('lr_decay_rate', None, 'If supplied, use ExponentialDecay.')
            p.define('lr_decay_iters', None, 'If supplied, use ExponentialDecay.')
            p.define('lr_piecewise', None, 'If supplied, use Piecewise.  Example:' \
                                           '[(0, 1e-3), (200, 1e-4)].')

            # p.define('use_adaptive_lr', None, 'Experimental.')
            # p.define('use_adaptive_lr_decay_to_zero', None, 'Experimental.')

            p.define('final_decay_rate', None, 'Experimental.')
            p.define('linear_decay_to_zero', False,
                     'Linearly decay from lr to 0 in val_iters.')
            p.define('reduce_lr_within_val_iter', False,
                     'Reduce LR within each val iter?')

            # Training.
            p.define('inherit_optimizer_state', False, 'Experimental.  For Adam.')
            p.define('epochs', 1, 'Num epochs to train.')
            p.define('bs', 1024, 'Batch size.')
            p.define('val_iters', 500, '# of value iterations.')
            p.define('increment_iter_despite_timeouts', False,
                     'Increment the iteration counter even if timeouts occurred?')
            # p.define('loss_type', None, 'Options: None (MSE), mean_qerror.')
            p.define('cross_entropy', False, 'Use cross entropy loss formulation?')
            p.define('l2_lambda', 0, 'L2 regularization lambda.')
            p.define('adamw', None,
                     'If not None, the weight_decay param for AdamW.')
            p.define('label_transforms', ['log1p', 'standardize'],
                     'Transforms for labels.')
            # p.define('label_transform_running_stats', False,
            #          'Use running mean and std to standardize labels?' \
            #          '  May affect on-policy.')
            p.define('update_label_stats_every_iter', True,
                     'Update mean/std stats of labels every value iteration?  This' \
                     'means the scaling of the prediction targers will shift.')
            p.define('gradient_clip_val', 0, 'Clip the gradient norm computed over' \
                                             ' all model parameters together. 0 means no clipping.')
            p.define('early_stop_on_skip_fraction', None,
                     'If seen plans for x% of train queries produced, early stop.')
            # Validation.
            p.define('validate_fraction', 0.1,
                     'Sample this fraction of the dataset as the validation set.  ' \
                     '0 to disable validation.')
            p.define('validate_every_n_epochs', 5,
                     'Run validation every this many training epochs.')
            p.define(
                'validate_early_stop_patience', 3,
                'Number of validations with no improvements before early stopping.' \
                '  Thus, the maximum # of wasted train epochs = ' \
                'this * validate_every_n_epochs).'
            )
            # Testing.
            p.define('test_every_n_iters', 1,
                     'Run test set every this many value iterations.')
            p.define('test_after_n_iters', 0,
                     'Start running test set after this many value iterations.')
            p.define('test_using_retrained_model', False,
                     'Whether to retrain a model from scratch just for testing.')
            p.define('track_model_moving_averages', False,
                     'Track EMA/SWA of the agent?')
            p.define('ema_decay', 0.95, 'Use an EMA model to evaluate on test.')

            # Pre-training.
            p.define('sim', True, 'Initialize from a pre-trained SIM model?')
            p.define('finetune_out_mlp_only', False, 'Freeze all but out_mlp?')

            p.define(
                'sim_checkpoint', "best_model.pth",
                'Path to a pretrained SIM checkpoint.  Load it instead '
                'of retraining.')

            p.define(
                'param_noise', 0.0,
                'If non-zero, add Normal(0, std=param_noise) to Linear weights ' \
                'of the pre-trained net.')
            p.define(
                'param_tau', 1.0,
                'If non-zero, real_model_t = tau * real_model_tm1 + (1-tau) * SIM.')
            p.define(
                'use_ema_source', False,
                'Use an exponential moving average of source networks?  If so, tau' \
                ' is used as model_t := source_t :=' \
                ' tau * source_(t-1) + (1-tau) * model_(t-1).'
            )
            p.define(
                'skip_sim_init_iter_1p', False,
                'Starting from the 2nd iteration, skip initializing from ' \
                'simulation model?'
            )
            p.define(
                'generic_ops_only_for_min_card_cost', False,
                'This affects sim model training and only if MinCardCost is used. ' \
                'See sim.py for documentation.')
            p.define(
                'sim_data_collection_intermediate_goals', True,
                'This affects sim model training.  See sim.py for documentation.')

            # Training data / replay buffer.
            p.define(
                'init_baseline_experience', 'initial_policy_data.pkl',
                'Initial data set of query plans to learn from. By default, this' \
                ' is the expert optimizer experience collected when baseline' \
                ' performance is evaluated.'
            )
            p.define('skip_training_on_expert', True,
                     'Whether to skip training on expert plan-latency pairs.')
            p.define(
                'dedup_training_data', True,
                'Whether to deduplicate training data by keeping the best cost per' \
                ' subplan per template.'
            )
            p.define('on_policy', False,
                     'Whether to train on only data from the latest iteration.')
            p.define(
                'use_last_n_iters', -1,
                'Train on data from this many latest iterations.  If on_policy,' \
                ' this flag is ignored and treated as 1 (latest iter).  -1 means' \
                ' train on all previous iters.')
            p.define('skip_training_on_timeouts', False,
                     'Skip training on executions that were timeout events?')
            p.define(
                'use_new_data_only', False,
                'Experimental; has effects if on_policy or use_last_n_iters > 0.' \
                '  Currently only implemented in the dedup_training_data branch.')
            p.define(
                'per_transition_sgd_steps', -1, '-1 to disable.  Takes effect only' \
                                                ' for when p.use_last_n_iters>0 and p.epochs=1.  This controls the' \
                                                ' average number of SGD updates taken on each transition.')
            p.define('physical_execution_hindsight', False,
                     'Apply hindsight labeling to physical execution data?')
            p.define(
                'replay_buffer_reset_at_iter', None,
                'If specified, clear all agent replay data at this iteration.')

            # Offline replay.
            p.define(
                'prev_replay_buffers_glob', None,
                'If specified, load previous replay buffers and merge them as training purpose.'
            )
            p.define(
                'prev_replay_buffers_glob_val', None,
                'If specified, load previous replay buffers and merge them as validation purpose.'
            )

            p.define(
                'agent_checkpoint', None,
                'Path to a pretrained agent checkpoint.  Load it instead '
                'of retraining.')
            p.define('prev_replay_keep_last_fraction', 1,
                     'Keep the last fraction of the previous replay buffers.')

            # Modeling: tree convolution (suggested).
            # p.define('tree_conv', True, "If true, use tree convolutional neural net.')
            # p.define('tree_conv_version', None, 'Options: None.')

            p.define('sim_query_featurizer', True,
                     'If true, use SimQueryFeaturizer to produce query features.')
            # Featurization.
            p.define('perturb_query_features', None,
                     'If not None, randomly perturb query features on each forward' \
                     ' pass, and this flag specifies ' \
                     '(perturb_prob_per_table, [scale_min, scale_max]).  ' \
                     'A multiplicative scale is drawn from ' \
                     'Unif[scale_min, scale_max].  Only performed when training ' \
                     'and using a query featurizer with perturbation implemented.')

            # Modeling: Transformer (deprecated).  Enabled when tree_conv is False.
            p.define('v2', True, 'If true, use TransformerV2.')
            p.define('pos_embs', True, 'Use positional embeddings?')
            p.define('dropout', 0.0, 'Dropout prob for transformer stack.')

            # Inference.
            p.define('check_hint', True, 'Check hints are respected?')

            # p.define('beam', 20, 'Beam size.')
            # p.define(
            #     'search_method', 'beam_bk',
            #     'Algorithm used to search for execution plans with cost model.')
            # p.define(
            #     'search_until_n_complete_plans', 10,
            #     'Keep doing plan search for each query until this many complete' \
            #     ' plans have been found.  Returns the predicted cheapest one out' \
            #     ' of them.  Recommended: 10.')

            # p.define('planner_config', None, 'See optimizer.py#PlannerConfig.')

            p.define(
                'avoid_eq_filters', False,
                'Avoid certain equality filters during planning (required for Ext-JOB).'
            )

            p.define('sim_use_plan_restrictions', True, 'Experimental.')
            p.define('real_use_plan_restrictions', True, 'Experimental.')

            # Exploration during inference.
            p.define(
                'epsilon_greedy', 0,
                'Epsilon-greedy policy: with epsilon probability, execute a' \
                ' randomly picked plan out of all complete plans found, rather' \
                ' than the predicted-cheapest one out of them.')
            p.define('epsilon_greedy_random_transform', False,
                     'Apply eps-greedy to randomly transform the best found plan?')
            p.define('epsilon_greedy_random_plan', False,
                     'Apply eps-greedy to randomly pick a plan?')
            p.define('epsilon_greedy_within_beam_search', False,
                     'Apply eps-greedy to within beam search?')
            p.define('explore_soft_v', False,
                     'Sample an action from the soft V-distribution?')
            p.define('explore_visit_counts', False, 'Explores using a visit count?')
            p.define('explore_visit_counts_sort', False,
                     'Explores by executing the plan with the smallest ' \
                     '(visit count, predicted latency) out of k-best plans?')
            p.define('explore_visit_counts_latency_sort', False,
                     'Explores using explore_visit_counts_sort if there exists ' \
                     'a plan that has a 0 visit count. Else sorts by predicted latency.')

            p.define('special_timeout_label', True,
                     'Use a constant timeout label (4096 sec)?')

            # Safe execution.
            # p.define('use_timeout', True, 'Use a timeout safeguard?')
            # p.define('initial_timeout_ms', None, 'Timeout for iter 0 if not None.')
            # p.define('timeout_slack', 2,
            #          'A multiplier: timeout := timeout_slack * max_query_latency.')
            # p.define('relax_timeout_factor', None,
            #          'If not None, a positive factor to multiply with the current' \
            #          ' timeout when relaxation conditions are met.')
            # p.define('relax_timeout_on_n_timeout_iters', None,
            #          'If there are this many timeout iterations up to now, relax' \
            #          ' the current timeout by relax_timeout_factor.')

            # Execution.
            # p.define('use_local_execution', False,
            #          'For query executions, connect to local engine or the remote' \
            #          ' cluster?  Non-execution EXPLAINs are always issued to' \
            #          ' local.')

            # p.define('use_cache', True, 'Skip executing seen plans?')

            p.define('model_save_path', './',
                     'Where to save the model ')

            p.define('model_prefix', '',
                     'Model prefix, for naming the model')
            return p


    def _train(dbname: str, db_path: str, config_none):
        buffer_mngr = BufferManager(db_path)
        config = BalsaParams.Params()

        # Create and train expert
        with PostgresConnector(dbname) as conn:
            expert = GraphOptimizerExpert(params=config, db_cli=conn)

            # training_data = buffer_mngr.storage.get_hybrid_qo_training_data(n_samples)
            # if not training_data:
            #     raise ValueError("No training data available for Bao")

            # Train from unified data
            expert.Run()
            # expert.train(training_data)
            # expert.save()
            print("Training completed!")
            print("Save expert training completed successfully!")


    # def _predict(db_path: str, input_sql_dir: str, dbname: str, config: GraphExpertConfig):
    #     buffer_mngr = BufferManager(db_path)
    #
    #     query_w_name = read_sql_files(input_sql_dir)
    #     with PostgresConnector(dbname) as conn:
    #         expert = GraphOptimizerExpert(config=config, db_cli=conn)
    #         expert.load()
    #         for sql_name, sql in query_w_name:
    #             query_id = sql_name
    #             new_sql_str, predicted_time_ms = expert.predict(query_id, sql, conn)
    #
    #             # Execute the sql with hint, save
    #             execution_id = buffer_mngr.collect_query_execution_history(
    #                 conn=conn, sql=new_sql_str, query_id=query_id, hints=None, hint_id=-1)
    #
    #             print(f"✓ Saved execution {execution_id} for {query_id}: {sql[:60]}...")
    #             print(execution_id)

    n_samples = 20

    parser = argparse.ArgumentParser(description="Train Bao Expert from Unified Data")
    parser.add_argument('--dbname', type=str, default='imdb_ori', help='Database name')
    parser.add_argument("--db_path", default="buffer_imdb_ori_db", help="Path to unified SQLite database")
    parser.add_argument(
        '--input_sql_dir',
        type=str,
        default="/Users/kevin/project_python/AI4QueryOptimizer/experiment_setup/workloads/bao/join_unique",
        help='Input SQL dir (one query per file)')
    args = parser.parse_args()

    # ---------- JOB ----------
    # config_job = GraphExpertConfig(
    #     queries_file="workload/query_join__train.json",
    # )

    _train(dbname=args.dbname, db_path=args.db_path, config_none=None)
    # _predict(args.db_path, args.input_sql_dir, args.dbname, config=config_job)
