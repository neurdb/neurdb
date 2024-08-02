# NeurDB
This is NeurDB.

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

### Install nr

```bash
docker exec -it neurdb_dev bash
```

### Install nr_inference

```bash
docker exec -it neurdb_dev bash
```

## Test Extension

```bash
$NEURDBPATH/psql/bin/psql  -h localhost -U postgres -d postgres -p 5432
```

Run extension

```sql
DROP EXTENSION neurdb_extension;
CREATE EXTENSION neurdb_extension;

SELECT mlp_clf('class', 'iris', '', '/code/neurdb-dev/contrib/nr/pysrc/config.ini');

PREDICT VALUE OF
  class
FROM
  iris;
```

# Debug

## Debug PostgreSQL

```bash
# users
su - postgres
```

Chaning the postgresql code, then restart the pg server

```bash
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data stop

make
make install
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data -l logfile start
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



## Debug Rust Extensin for training

After updating the codebase, run the following

```bash
cargo test -- --nocapture
```

Once the test pass, re install extension

```bash
cd /code/neurdb-dev/contrib/nr
cargo pgrx install --pg-config $NEURDBPATH/psql/bin/pg_config --release
```

Then

```bash
$NEURDBPATH/psql/bin/psql  -h localhost -U postgres -d postgres -p 5432
```

```sql
DROP EXTENSION neurdb_extension;
CREATE EXTENSION neurdb_extension;
```

## Debug C Extension for inference
Start the PostgreSQL server

```bash
$NEURDBPATH/psql/bin/psql  -h localhost -U postgres -d postgres -p 5432
```

Drop/Create the extension

```sql
DROP EXTENSION pg_model;
CREATE EXTENSION pg_model;
```


# Some CMDs

Set global path of the neurdb

```bash
NEURDBPATH=/code/neurdb-dev
```

Debug if there is an error

```bash
# check if the .sql file generated in the folder psql/share/extension/
# if it's empty, then run
cargo pgrx schema --pg-config ./psql/bin/pg_config --features python --release

# if the PGRX_INCLUDEDIR_SERVER and PGXS cannot found, then run
PGRX_INCLUDEDIR_SERVER=$NEURDBPATH/psql/include/postgresql/server PGXS=$NEURDBPATH/psql/lib/postgresql/pgxs/src/makefiles/pgxs.mk cargo pgrx install --pg-config $NEURDBPATH/psql/bin/pg_config --release

```

```bash
chown $(whoami) $NEURDBPATH/psql/data
```

```bash
export LANG="en_US.UTF-8"
export LC_COLLATE="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export LC_MESSAGES="en_US.UTF-8"
export LC_MONETARY="en_US.UTF-8"
export LC_NUMERIC="en_US.UTF-8"
export LC_TIME="en_US.UTF-8"
export DYLD_LIBRARY_PATH=$NEURDBPATH/psql/lib:$DYLD_LIBRARY_PATH
```

