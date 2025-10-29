# Third-party imports
import torch


class BaseConfig:
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    LOG_DIR = "./logs"
    RESULT_DATA_BASE = "./experiment_result/result_data/"

    @staticmethod
    # --- Utility Functions ---
    def query_file_to_ident(file_name: str) -> str:
        """Convert query filename to identifier (e.g., '1a.sql' -> '01a')."""
        ident = file_name.split('.sql')[0]
        return f"{ident[:-1].zfill(2)}{ident[-1]}"

    # --- Utility Functions ---
    @staticmethod
    def query_file_to_ident_stack(file_name: str) -> str:
        """Convert query filename to identifier (e.g., '1a.sql' -> '01a')."""
        ident = file_name.split('.sql')[0]
        return ident

    @staticmethod
    def ident_to_sql_filename(query_ident: str) -> str:
        """Convert query identifier back to SQL filename (e.g., '01a' -> '1a.sql')."""
        num = str(int(query_ident[:-1]))  # Remove leading zeros and get number
        letter = query_ident[-1]
        return f"{num}{letter}.sql"
