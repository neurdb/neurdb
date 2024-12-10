

## Usage

### Create Shell

```bash
docker exec -it neurdb_dev bash
```

### Run Connector (in Docker Container)

```bash
$NEURDBPATH/psql/bin/psql -h 0.0.0.0
```

<!--
### Run tests

> [!NOTE]
> In the current state, the implementation of `PREDICT` syntax is not complete but scheduled. Once it is done, you can use the following syntax to run the training/inference on the specific data table, e.g.,
> ```
> PREDICT CLASS OF class FROM iris;
> ```
-->

### Start/stop the server (in Docker container)

```bash
# (Start/stop as a service)
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data start
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data stop

# (Start in frontend)
$NEURDBPATH/psql/bin/postgres -D $NEURDBPATH/psql/data -h 0.0.0.0
```



## Development

### Code Formatting

All tools are managed by pre-commit.

After installing pre-commit, run the following command:

```bash
pre-commit run -a
```

### Increase logging level

Update `$NEURDBPATH/psql/data/postgresql.conf` by changing the corresponding settings:

```ini
log_min_messages = DEBUG1
log_min_error_statement = DEBUG1
```

### Miscellaneous

If the server complains about the permission of the data directory, run the following command to fix it:

```bash
chmod 750 $NEURDBPATH/psql/data
```

Sometimes, the OS language settings could also affect the server starting. In such case, run the following commands:

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

## Troubleshooting

### Building

#### `./configure: line ...: config.log: Permission denied`

Check the permissions of (1) executing for `dbengine/configure` (2) writing for other directories.

#### `pg_ctl: directory "/code/neurdb-dev/psql/data" is not a database cluster directory`

This happens when you have directory `psql/data` without initializing the database, e.g., if the init script exits abnormally before. By default, to avoid data loss, the init script will not touch `psql/data` if it already exists. To solve this, remove `psql/data` manually.

# Build DBEngine

```bash
# users
su - postgres
make clean

# stop existing db
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data stop

# build and restart
./configure --prefix=$NEURDBPATH/psql --enable-debug
make
make install

mkdir -p $NEURDBPATH/psql/data
sudo chown $(whoami) $NEURDBPATH/psql/data
$NEURDBPATH/psql/bin/initdb -D $NEURDBPATH/psql/data
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data -l logfile start
$NEURDBPATH/psql/bin/psql  -h localhost -U postgres -d postgres -p 5432

# restart
$NEURDBPATH/psql/bin/pg_ctl reload -D $NEURDBPATH/psql/data
```

If there are any error, check the log at

```bash
# Update the `psql/postgresql.conf` by adding those two lines
log_min_messages = DEBUG1
log_min_error_statement = DEBUG1


# then check the log file
./logfile
```

## Buiild extension

Stop DB, recompile the extension, start DB.

```bash
# For debug, build extension with
make debug && make install
```



## Usage

```sql
CREATE TABLE IF NOT EXISTS frappe_test (
  click_rate INT,
  feature1 INT,
  feature2 INT,
  feature3 INT,
  feature4 INT,
  feature5 INT,
  feature6 INT,
  feature7 INT,
  feature8 INT,
  feature9 INT,
  feature10 INT
);
```

```bash
psql -h localhost -U postgres -d postgres -c "\copy frappe_test FROM './frappe.csv' DELIMITER ',' CSV HEADER;"
```

```sql
PREDICT VALUE OF click_rate
FROM frappe_test
TRAIN ON feature1, feature2;
```



# Debug PG

Inside docker container,

```bash
apt-get update && apt-get install -y gdb
apt-get update && apt-get install -y gdbserver
apt-get update && apt-get install -y net-tools
sudo apt install lsof -y


gdbserver 0.0.0.0:1234 --attach 9948

```



# Useful references

[Hook Functions](https://github.com/taminomara/psql-hooks/blob/master/Detailed.md)

[Hook Examples](https://wiki.postgresql.org/images/e/e3/Hooks_in_postgresql.pdf)
