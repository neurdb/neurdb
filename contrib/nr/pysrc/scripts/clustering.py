import argparse
import csv
import time

from models import build_model
from shared_config.config import parse_config_arguments
import torch

if __name__ == "__main__":

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Train an ARMNet model")
    parser.add_argument("--model_path", type=str, default="./avazue_model.pt", help="Path to save the model")
    parser.add_argument('--input_file', type=str, default="./ava_res.csv", help='Path to the input CSV file')
    parser.add_argument('--output_folder', type=str, default="./",
                        help='Path to the output folder where cluster files will be saved')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters for K-Means')
    parser.add_argument('--top_clusters', type=int, default=4, help='Number of top largest clusters to save')

    # Parse arguments
    args = parser.parse_args()

    # Parse configuration
    config_args = parse_config_arguments("/Users/kevin/project_c++/neurdb_proj/neurdb-dev/contrib/nr/pysrc/config.ini")
    config_args.nfield = 22
    config_args.nfeat = 1544272

    # Initialize and load the model
    builder = build_model("armnet", config_args)
    builder._init_model_arch()
    builder._model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    # Reading the CSV file without pandas and without a header
    data = []
    with open(args.input_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)

    # Assuming all columns are used for embedding except the first one
    embeddings = []
    for row in data:
        x = {"id": torch.tensor([int(item) for item in row[1:]], dtype=torch.long),
             "value": torch.ones((len(data), config_args.nfield), dtype=torch.float)
             }
        emb = builder._model.embedding(x)
        embeddings.append(emb)
        print(embeddings)
        time.sleep(1000)
