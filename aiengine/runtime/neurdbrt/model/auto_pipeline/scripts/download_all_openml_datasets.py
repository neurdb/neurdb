import openml
from tqdm import tqdm

if __name__ == "__main__":
    datasets_df = openml.datasets.list_datasets(
        status="active", output_format="dataframe"
    )
    ids = datasets_df["did"].values.tolist()
    print(f"len(ids)={len(ids)}, ids[-1]={ids[-1]}")

    for id in tqdm(ids, total=len(ids)):
        try:
            openml.datasets.get_dataset(dataset_id=id)
        except Exception as e:
            print(f"ERROR: id {id} failed with error: {e}")
