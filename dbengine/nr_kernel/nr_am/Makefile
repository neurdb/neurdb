NEURDBPATH=/code/neurdb-dev
EXTPATH = /code/neurdb-dev/dbengine/nr_kernel/nr_am
MODULE_big = nram
OBJS = ${EXTPATH}/src/nram.o


EXTENSION = nram
DATA = sql/nram--1.0.sql
PGFILEDESC = "nram - CCaaLF custom table access method"

# Link to RocksDB C API
PG_CPPFLAGS += -I/usr/local/include
SHLIB_LINK += -L/usr/local/lib -lrocksdb

REGRESS =

PG_CONFIG = ${NEURDBPATH}/psql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)


debug:
	$(MAKE) -f Makefile PG_CPPFLAGS="$(PG_CPPFLAGS) -DDEBUG" SHLIB_LINK="$(SHLIB_LINK)" CFLAGS="-g -O0 -Wall"
