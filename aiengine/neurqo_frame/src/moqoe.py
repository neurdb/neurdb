# Standard library imports
import base64
import zlib
from typing import Dict, Optional

# Third-party imports
import torch

# Local/project imports
from common import BaseConfig, get_config
from db.pg_conn import PostgresConnector
from exp_buffer.buffer_mngr import BufferManager
from expert_pool.hint_plan_sel_expert import TreeCNNConfig, TreeCNNRegExpert
from expert_pool.join_order_expert import MCTSConfig, MCTSOptimizerExpert
from expert_router.controller_offline import ModelBuilder

# Constants
DEFAULT_NUM_COLUMNS = 108
DEFAULT_NUM_HEADS = 4
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_NUM_LAYERS = 2
ROUTER_MODEL_NAME = "hypered_2"
DEFAULT_ONLINE_UPDATE_TRIGGER = 10
DEFAULT_TRAIN_DATA_LIMIT = 100


class RoutingHelper:
    """
    Router that selects the best optimizer expert for a given SQL query.

    Uses a learned routing model to predict which expert (optimizer) will perform
    best for a given query based on query features and historical performance.
    """

    def __init__(self, cfg: BaseConfig, model_path: str, dataset: str):
        """
        Initialize the routing helper.

        Args:
            cfg: Base configuration object
            model_path: Path to the directory containing router models
            dataset: Dataset name (e.g., 'imdb', 'stack')
        """
        self.cfg = cfg
        self.model_builder = ModelBuilder(
            num_columns=DEFAULT_NUM_COLUMNS,
            output_dim=len(cfg.ALL_METHODS),
            model_path_prefix=f"{model_path}/router_model.pth",
            embedding_path=None,
            num_heads=DEFAULT_NUM_HEADS,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            is_fix_emb=None,
            num_layers=DEFAULT_NUM_LAYERS,
            dataset=dataset,
            cfg=cfg,
        )
        self.model_builder.load_model(ROUTER_MODEL_NAME)
        self.id_to_method = {
            idx: method for method, idx in cfg.FIXED_LABEL_MAPPING.items()
        }

    def optimizer_selection(self, sql: str, db_cli: PostgresConnector) -> list:
        """
        Select the best optimizer(s) for a given SQL query.

        Args:
            sql: SQL query string
            db_cli: PostgreSQL connector instance

        Returns:
            List of optimizer method names, ranked by predicted performance
        """
        query_id = RoutingHelper._encode_sql_to_id(sql)
        optimizer_idx = self.model_builder.inference_single_sql(
            sql=sql, query_id=query_id, db_cli=db_cli
        )

        print(f"Selected optimizer index: {optimizer_idx}")
        res_rank = [self.id_to_method[idx] for idx, _ in optimizer_idx]
        print(f"Mapped method: {res_rank}")
        return res_rank

    @staticmethod
    def online_update(performance: Optional[dict] = None) -> None:
        # Placeholder for learning a routing policy from performance stats
        return

    @staticmethod
    def _map_back_to_method(predictions: dict, cfg: BaseConfig) -> dict:
        id_to_method = {idx: method for method, idx in cfg.FIXED_LABEL_MAPPING.items()}
        predicted_methods = {
            query: id_to_method[pred] for query, pred in predictions.items()
        }
        return predicted_methods

    @staticmethod
    def _encode_sql_to_id(sql_string: str) -> str:
        compressed = zlib.compress(sql_string.encode("utf-8"))
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")
        return encoded


class MoQOEController:
    """
    Main controller for Mixture of Query Optimizer Experts (MoQOE).

    Manages a pool of expert optimizers and routes queries to the best expert
    based on learned routing decisions. Supports online learning and expert
    fine-tuning.
    """

    def __init__(
        self,
        buffer_path: str,
        config: BaseConfig,
        DATASET: str = "imdb",
        database_name: str = "imdb_ori",
        model_path: str = "./models/router_models",
        online_update_trigger: int = DEFAULT_ONLINE_UPDATE_TRIGGER,
    ):
        """
        Initialize MoQOE controller.

        Args:
            buffer_path: Path to the buffer database file
            config: Base configuration object
            DATASET: Dataset name (e.g., 'imdb', 'stack')
            database_name: PostgreSQL database name
            model_path: Path to model directory
            online_update_trigger: Number of queries before triggering online training
        """
        self.config = config
        self.buffer_mngr = BufferManager(buffer_path)
        self.routing_network = RoutingHelper(
            cfg=self.config, model_path=model_path, dataset=DATASET
        )

        self.db_cli = PostgresConnector(database_name)
        self.db_cli.connect()
        self.current_database: Optional[str] = None

        # Expert pool management
        self.expert_pools: Dict[str, object] = {}
        self.expert_chosen_count: Dict[str, int] = {}
        self.online_update_sql_trigger = online_update_trigger

        self.load_experts()

    def load_experts(self) -> None:
        """
        Load and initialize all expert optimizers.

        Currently loads:
        - HintPlanSel: Expert for hint plan selection
        - JoinOrder: Expert for join order optimization
        """
        # Configure HintPlanSel expert
        hint_plan_sel_expert_config = TreeCNNConfig(
            train_epochs=100, batch_size=16, device=torch.device("cpu")
        )
        hint_plan_sel_expert = TreeCNNRegExpert(
            config=hint_plan_sel_expert_config, buffer_mngr=self.buffer_mngr
        )
        hint_plan_sel_expert.load()
        self._add_expert("HintPlanSel", hint_plan_sel_expert)

        # Configure JoinOrder expert
        join_order_expert_config = MCTSConfig(
            max_alias_num=40,
            max_column=100,
            id2aliasname=self.config.id2aliasname,
            aliasname2id=self.config.aliasname2id,
        )
        join_order_expert = MCTSOptimizerExpert(
            config=join_order_expert_config,
            buffer_mngr=self.buffer_mngr,
            db_cli=self.db_cli,
        )
        join_order_expert.load()
        self._add_expert("JoinOrder", join_order_expert)

        print(
            "\n === [MoQOEController.load_experts] Experts loaded successfully === \n"
        )

    def _online_train(self, e_id: str) -> None:
        """
        Fine-tune a single expert online using recent query performance data.

        Args:
            e_id: Expert identifier
        """
        expert = self._get_expert_by_id(e_id)
        if expert is None:
            print(f"Warning: Expert {e_id} not found, skipping online training")
            return

        print(f"\n === [MoQOEController._online_train] Training expert: {e_id} === \n")
        performance = expert.train_and_save(train_data_limit=DEFAULT_TRAIN_DATA_LIMIT)

        # Update router based on expert performance feedback
        self.routing_network.online_update(performance)

    def inference(self, sql: str) -> tuple:
        """
        Optimize a SQL query by routing to the best expert.

        The process:
        1. Route query to best expert using routing network
        2. Get optimized SQL from selected expert
        3. Track expert usage and trigger online training when needed

        Args:
            sql: SQL query string to optimize

        Returns:
            Tuple of (optimized_sql, expert_name)
        """
        if not self.expert_pools:
            print("Warning: No experts available, returning original SQL")
            return (sql, "cost-based optimizer")  # Fallback to cost-based optimizer

        # Route query to best expert
        try:
            exp_rank = self.routing_network.optimizer_selection(
                sql=sql, db_cli=self.db_cli
            )
        except Exception as e:
            print(
                f"Warning: Error in optimizer selection: {e}, falling back to cost-based optimizer"
            )
            return (
                sql,
                "cost-based optimizer",
            )  # Fallback to cost-based optimizer on error

        # Select expert with highest rank (lower index = higher rank)
        selected_exp_id = min(
            (x for x in self.expert_pools.keys() if x in exp_rank),
            key=lambda x: exp_rank.index(x),
            default=None,
        )

        if selected_exp_id is None:
            print("Warning: No expert selected, returning original SQL")
            return (sql, "cost-based optimizer")  # Fallback to cost-based optimizer

        selected_expert = self.expert_pools[selected_exp_id]
        print(
            f"\n === [MoQOEController.inference] Selected expert: {selected_exp_id} === \n"
        )

        # Get optimized SQL from expert
        try:
            updated_sql, expert_hint = selected_expert.sql_enhancement(
                sql=sql, db_cli=self.db_cli
            )
            print(
                f"\n === [MoQOEController.inference] Optimized SQL: {updated_sql} === \n"
            )
            print(f" === [MoQOEController.inference] Expert hint: {expert_hint} === \n")

            # Update usage counter and trigger online training if needed
            self.expert_chosen_count[selected_exp_id] = (
                self.expert_chosen_count.get(selected_exp_id, 0) + 1
            )
            if (
                self.expert_chosen_count[selected_exp_id]
                % self.online_update_sql_trigger
                == 0
            ):
                self._online_train(selected_exp_id)

            return (updated_sql, selected_exp_id)
        except Exception as e:
            print(
                f"Warning: Error in expert {selected_exp_id}: {e}, falling back to cost-based optimizer"
            )
            return (
                sql,
                "cost-based optimizer",
            )  # Fallback to cost-based optimizer on error

    # ---------- Expert Pool Management ----------

    def _add_expert(self, e_id: str, expert: object) -> None:
        """
        Add or replace an expert in the pool.

        Args:
            e_id: Expert identifier
            expert: Expert instance
        """
        self.expert_pools[e_id] = expert
        self.expert_chosen_count.setdefault(e_id, 0)

    def _remove_expert(self, e_id: str) -> None:
        """
        Remove an expert from the pool.

        Args:
            e_id: Expert identifier
        """
        self.expert_pools.pop(e_id, None)
        self.expert_chosen_count.pop(e_id, None)

    def _get_expert_by_id(self, e_id: str) -> Optional[object]:
        """
        Get expert by ID.

        Args:
            e_id: Expert identifier

        Returns:
            Expert instance or None if not found
        """
        return self.expert_pools.get(e_id)


def main():
    # Configuration
    DATASET = "imdb"

    TEST_SQL = """
    SELECT MIN(mc.note) AS production_note,
           MIN(t.title) AS movie_title,
           MIN(t.production_year) AS movie_year
    FROM company_type AS ct,
         info_type AS it,
         movie_companies AS mc,
         movie_info_idx AS mi_idx,
         title AS t
    WHERE ct.kind = 'production companies'
      AND it.info = 'bottom 10 rank'
      AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
      AND t.production_year >2000
      AND ct.id = mc.company_type_id
      AND t.id = mc.movie_id
      AND t.id = mi_idx.movie_id
      AND mc.movie_id = mi_idx.movie_id
      AND it.id = mi_idx.info_type_id;

    """
    cfg = get_config(DATASET)
    print("Initializing RoutingHelper...")
    routing_cntr = MoQOEController(
        buffer_path="./models/buffer_imdb_ori.db",
        config=cfg,
        DATASET="imdb",
        database_name="imdb_ori",
    )
    res = routing_cntr.inference(sql=TEST_SQL)
    print(res)


if __name__ == "__main__":
    main()
