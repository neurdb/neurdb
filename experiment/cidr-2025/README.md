# CIDR-2025 Experiments

This folder contains the code and data for the experiments in the paper "XXX" submitted to CIDR 2025.

## Dataset

### Avazu

The Click-Through Rate Prediction dataset contains predictions of whether a mobile ad will be clicked. The dataset is available at [Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction/data). 

The dataset contains **40428967** samples. It has XXX columns, including the label column.

### Criteo

The dataset contains **45840617** samples. We take the first 13 features as the input features.


## Get Started

Clone the repository:
```bash
git clone https://github.com/neurdb/neurdb-dev.git
```

Change the permission of the project folder:
```bash
cd neurdb-dev
chmod 777 -R .
```

### Build Dockerfile
In the server, build the docker image. Remember to `cd` to the `deploy` folder before running the build script, otherwise the docker bind mount will fail.
```bash
cd ~/neurdb-dev/deploy
bash build.sh
```

Go into the docker environment:
```bash
docker exec -it neurdb_dev bash
```

### Prepare Data
Navigate to the experiment folder:
```bash
cd $NEURDBPATH/experiment/cidr-2025
```

Add necessary modules:
```bash
export PYTHONPATH=$NEURDBPATH/contrib/nr/pysrc:$PYTHONPATH
```

Run prepare_data.py script:
```bash
python3 prepare_data.py --dataset_name dataset_name --input_file /path/to/data.libsvm --file_type libsvm --random_state 10
```

`--dataset_name` name of the dataset (e.g., frappe)

`--input_file` path to the input file, in libsvm, npy, or csv format (e.g., /path/to/data.libsvm)

`--file_type` type of the input file, either libsvm, npy, or csv (e.g., libsvm)

`--random_state` used to shuffle the data

### Start Python Server
```bash
cd $NEURDBPATH/contrib/nr/pysrc
python3 app.py
```

### Run NeurDB
Connect to the database:
```bash
$NEURDBPATH/psql/bin/psql  -h localhost -U postgres -d postgres -p 5432
```

Train a model:
```postgresql
SELECT nr_train('armnet', 'frappe', 10, ARRAY['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10'], 'label');
```

Forward inference:
```postgresql
SELECT nr_inference('armnet', 1, 'frappe', 10, ARRAY['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10']);
```

Finetune a model:
```postgresql
SELECT nr_finetune('armnet', 1, 'frappe', 10, ARRAY['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10'], 'label');
```
