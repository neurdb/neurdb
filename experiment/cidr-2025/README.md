# CIDR-2025 Experiments

This folder contains the code and data for the experiments in the paper "XXX" submitted to CIDR 2025.

## Dataset

### Frappe

The frappe dataset contains a context-aware app usage log.  It consist of 96203 entries by 957 users for 4082 apps used in various contexts.

We sample 2 negative samples for 1 positive sample to create the dataset. The total number of samples is **288609**.

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
