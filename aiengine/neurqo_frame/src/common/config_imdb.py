# Standard library imports
import os

# Local/project imports
from common.base_config import BaseConfig


class IMDBConfig(BaseConfig):
    DB_NAME = "imdb_ori"
    ALL_METHODS = ['HintPlanSel', 'PlanGenSim', 'PlanGen', 'PostgreSQL', 'JoinOrder']
    FIXED_LABEL_MAPPING = {m: i for i, m in enumerate(ALL_METHODS)}
    EXECUTION_TIME_OUT = 360000.0

    DB_INFO_DICT = "./models/router_models/ori_table_info.json"
    QUERY_DIR = "./datasets/origin_datasets/imdb/"

    TRAIN_TEST = "./experiment_result/datasets/workload_data_train_test"
    EMBED_FILE = "./models/router_models/query_encodings_embedding_v2_imdb_ori.json"
    TRAIN_TEST_ONLINE_CONVARIATE = "./experiment_result/datasets/workload_data_train_test_online_mix_convariate"
    TESTPATH = "./experiment_result/datasets/workload_data_test"

    id2aliasname = {
        0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt',
        10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt',
        18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3',
        26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1',
        34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'
    }
    aliasname2id = {
        'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32,
        'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8,
        'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19,
        'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12,
        'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33
    }

    @staticmethod
    def load_sql_query(query_ident: str) -> str:
        sql_filename = IMDBConfig.ident_to_sql_filename(query_ident)
        filepath = os.path.join(IMDBConfig.QUERY_DIR, 'join-order-benchmark', sql_filename)
        with open(filepath) as f:
            return f.read().strip().replace("\n", " ")

    @staticmethod
    def get_all_queries():
        path = os.path.join(IMDBConfig.QUERY_DIR, 'join-order-benchmark')
        return sorted([q[:-4] for q in os.listdir(path) if q.endswith('.sql')])
