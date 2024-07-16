import re
import os
import torch
import random
import numpy as np


def seed_everything(seed=2022):
    '''
    [reference]
    https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_conditions(input_str):
    # Extract the field names
    fields = re.findall(r'\"([^\"]+)\"\s*\"([^\"]+)\"\)', input_str)
    # Extract the values
    values = re.findall(r':val\s*\"([^\"]+)\"', input_str)

    # Combine fields and values into a condition string
    conditions = []
    for (table, column), value in zip(fields, values):
        conditions.append(f"{table}.{column} = '{value}'")

    # Join all conditions with 'AND'
    return " AND ".join(conditions)
