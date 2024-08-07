#!/usr/bin

rm -rf /var/postgresql/data/*
initdb -D $PGDATA

echo "max_connections = 1000" >> $PGDATA/postgresql.conf
echo "shared_buffers = '4GB'" >> $PGDATA/postgresql.conf
echo "fsync = 'off'" >> $PGDATA/postgresql.conf
echo "full_page_writes = 'on'" >> $PGDATA/postgresql.conf
echo "max_locks_per_transaction = 128" >> $PGDATA/postgresql.conf
#echo "log_min_messages = debug5" >> $PGDATA/postgresql.conf

pg_ctl start -D $PGDATA
psql -d postgres -c 'create database ycsb';
psql -d postgres -c 'create database sysbench';
pg_ctl stop -D $PGDATA -m fast

