# CIDR-2025 Experiments

This folder contains the code and data for the experiments in the paper "XXX" submitted to CIDR 2025.

## Dataset

### Frappe

The frappe dataset contains a context-aware app usage log.  It consist of 96203 entries by 957 users for 4082 apps used in various contexts.

We sample 2 negative samples for 1 positive sample to create the dataset. The total number of samples is **288609**.

In our experiments, we randomly sample data from the dataset to create a larger dataset. The number of samples is specified by the `--num_rows` flag in the `prepare_data.py` script under the `experiment/cidr-2025` folder.

## Get Started

Clone the repository:
```bash
git clone https://github.com/neurdb/neurdb-dev.git
```

Change the permission of `meson.build`:
```bash
chmod 777 ~/neurdb-dev/meson.build
```

### Build Dockerfile
In the server, build the docker image:
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

Run prepare_data.py script:
```bash
python3 prepare_data.py --create_table --num_rows 10000
```

`--create_table` flag is used to create the table in the database if it does not exist.

`--num_rows` flag is used to specify the number of rows to be inserted into the table. (e.g., 10000)

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
