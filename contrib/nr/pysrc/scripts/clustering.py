import argparse
import csv
from sklearn.cluster import KMeans
import os

from models import build_model
from shared_config.config import parse_config_arguments
import torch


def generate_embeddings(args, config_args):
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
             "value": torch.ones((1, config_args.nfield), dtype=torch.float)}
        emb = builder._model.embedding(x)
        emb = emb.view(-1).tolist()
        embeddings.append(emb)

    return embeddings, data


def perform_kmeans(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(embeddings)
    return kmeans.labels_


def save_clusters(data, labels, args, top_clusters):
    cluster_mapping = {}
    for idx, label in enumerate(labels):
        if label in cluster_mapping:
            cluster_mapping[label].append(data[idx])
        else:
            cluster_mapping[label] = [data[idx]]

    # Sorting clusters by size and picking the top clusters
    sorted_clusters = sorted(cluster_mapping.items(), key=lambda x: len(x[1]), reverse=True)[:top_clusters]

    # Save each cluster's data into a separate CSV file
    for idx, (cluster_id, rows) in enumerate(sorted_clusters):
        output_path = os.path.join(args.output_folder, f"cluster_{cluster_id}.csv")
        print(f"cluster_{cluster_id}.csv has {len(rows)} rows")
        with open(output_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in rows:
                csvwriter.writerow(row)


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
    random_state = 0

    # Generate embeddings and original data
    embeddings, data = generate_embeddings(args, config_args)

    # Perform K-means clustering
    labels = perform_kmeans(embeddings, args.num_clusters)

    # Save clustered data
    save_clusters(data, labels, args, args.top_clusters)

"""
python script_name.py --model_path "/path/to/your/model.pt" --input_file "/path/to/your/input.csv" --output_folder "/path/to/output/directory" --num_clusters 5 --top_clusters 4
"""
