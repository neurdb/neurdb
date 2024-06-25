# Deployment on Server

## Clone Codes

In the server, use the following to clone the code, you may need to generate the token in Git Hub.

```bash
cd ~
git clone https://<username>:<token>@github.com/<username>/<repository>.git

# give docker write premission
chmod -R 777 ~/neurdb-dev
```

## Build Dockerfile

Build images

```bash
cd ~/neurdb-dev/deploy
bash build.sh
```

## Install Extensions

Install nr

```bash
docker exec -it neurdb_dev bash

# switch back to the root user with password: rootpassword
# su -

cd /code/neurdb-dev/contrib/nr
cargo pgrx init --pg16 $NEURDBPATH/psql/bin/pg_config
cargo clean
cargo pgrx install --pg-config $NEURDBPATH/psql/bin/pg_config --release
```

Install nr_inference

```bash
```

# Re-install
Folders created in docker can only removed from docker itself.
```bash
# switch back to the root user with password: rootpassword to remove 
su -
rm -rf $NEURDBPATH/contrib/nr/target

su - postgres
rm -rf $NEURDBPATH/psql
```
















# Dev Guide using docker

Build a Docker image and run the container

```bash
# (in the repository root folder)

docker build -t neurdb_dev .

mkdir venv

docker run -d --name neurdb_dev \
    -v .:/code/neurdb-dev \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v ./venv/:$HOME \
    -u $(id -u):$(id -g) \
    neurdb_dev

docker exec -it neurdb_dev bash

# install python libs
pip install -r contrib/nr/pysrc/requirement.txt

# install psycopg2
apt install python3-psycopg2
```

Then config DB in docker based on the doc 

https://github.com/neurdb/neurdb-docs/blob/main/dev-guide.md



# Load data

```
./psql/bin/psql -h localhost -U postgres -f ./dataset/iris_psql.sql
```



# Dependence (Already in Docker)

```bash
# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup install 1.78.0

# install cargo-pgrx
cargo install cargo-pgrx --version '0.11.4' --locked
```

```bash
# ensure rustc 1.78.0
rustc --version

# ensure cargo-pgrx v0.11.4
cargo-pgrx --version
```


# Install Extension Framework

Set global path of the neurdb

```bash
NEURDBPATH=/code/neurdb-dev
```

Install pgrx

```bash
# su - postgres
# Install new (or register existing) PostgreSQL installs.
# Make sure the neurdb is compiled already.
cd ./contrib/nr
cargo pgrx init --pg16 $NEURDBPATH/psql/bin/pg_config

# Config the right python version and corresponding dylib path in config.yaml, then run 
cargo clean
cargo pgrx install --pg-config $NEURDBPATH/psql/bin/pg_config --release
```

Debug if there is an error

```bash
# check if the .sql file generated in the folder psql/share/extension/
# if it's empty, then run 
cargo pgrx schema --pg-config ./psql/bin/pg_config --features python --release

# if the PGRX_INCLUDEDIR_SERVER and PGXS cannot found, then run 
PGRX_INCLUDEDIR_SERVER=$NEURDBPATH/psql/include/postgresql/server PGXS=$NEURDBPATH/psql/lib/postgresql/pgxs/src/makefiles/pgxs.mk cargo pgrx install --pg-config $NEURDBPATH/psql/bin/pg_config --release

```

# Usage

Start NeurDB with PGRX

```bash
cargo pgrx run --release
```

```sql
-- (In psql)

DROP EXTENSION neurdb_extension;
CREATE EXTENSION neurdb_extension;
```

Run linear regression example

```sql
SELECT mlp_clf('class', 'iris', '', '/code/neurdb-dev/contrib/nr/pysrc/config.ini');
```

Run PREDICT SQL example

```sql
\c neurdb

PREDICT VALUE OF
  class
FROM
  iris;
```


# Dev Extension

If the extension code is updated, then 

In psql command

```
DROP EXTENSION neurdb_extension;
exit
```

Rebuild the extension using

```bash
# run the test code 
cargo test

# If nothing run, install it
cargo pgrx install --pg-config $NEURDBPATH/psql/bin/pg_config --release
```

In psql command

```bash
./psql/bin/psql -d postgres
CREATE EXTENSION neurdb_extension;
```

# Dev Source Code

```bash
# user
su - postgres
```

Stop the current psql

```bash
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data stop
rm -rf $NEURDBPATH/psql/data
```

Rebuild with 

```bash
make 
make install
```

```bash
mkdir -p $NEURDBPATH/psql/data
chown $(whoami) $NEURDBPATH/psql/data
./psql/bin/initdb -D $NEURDBPATH/psql/data
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data -l logfile start
export LANG="en_US.UTF-8"
export LC_COLLATE="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export LC_MESSAGES="en_US.UTF-8"
export LC_MONETARY="en_US.UTF-8"
export LC_NUMERIC="en_US.UTF-8"
export LC_TIME="en_US.UTF-8"
export DYLD_LIBRARY_PATH=$NEURDBPATH/psql/lib:$DYLD_LIBRARY_PATH
./psql/bin/psql -d postgres

$NEURDBPATH/psql/bin/psql  -h localhost -U postgres -d postgres -p 5432
```

If there are any error, check the log at 

```bash
# Update the `psql/postgresql.conf` by adding those two lines
log_min_messages = DEBUG1
log_min_error_statement = DEBUG1


# then check the log file
./logfile
```

