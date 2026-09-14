# Standard library imports
import os

# Local/project imports
from common.base_config import BaseConfig


class StackConfig(BaseConfig):
    DB_NAME = "stack"
    ALL_METHODS = [
        "HintPlanSel",
        "PlanGenSim",
        "PlanGen",
        "PostgreSQL",
        "JoinOrder",
        "CostAdj",
    ]
    FIXED_LABEL_MAPPING = {m: i for i, m in enumerate(ALL_METHODS)}
    EXECUTION_TIME_OUT = 3 * 60 * 1000.0

    DB_INFO_DICT = "./models/router_models/ori_table_info_stack.json"
    QUERY_DIR = "./datasets/origin_datasets/stack/stack_queries"

    TRAIN_TEST = "./experiment_result/datasets/workload_data_train_test_stack"
    EMBED_FILE = "./models/router_models/query_encodings_embedding_v2_stack.json"
    TESTPATH = "./experiment_result/datasets/workload_data_test_stack"
    TRAIN_TEST_ONLINE_CONVARIATE = None

    @staticmethod
    def load_sql_query(query_ident: str) -> str:
        path = os.path.join(StackConfig.QUERY_DIR, f"{query_ident}.sql")
        with open(path) as f:
            lines = [
                l.strip() for l in f if l.strip() and not l.strip().startswith("--")
            ]
            return " ".join(lines)

    @staticmethod
    def get_all_queries():
        return sorted(
            [
                f.replace(".sql", "")
                for f in os.listdir(StackConfig.QUERY_DIR)
                if f.endswith(".sql")
            ]
        )
