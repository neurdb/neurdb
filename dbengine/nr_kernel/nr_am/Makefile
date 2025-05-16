NEURDBPATH=/code/neurdb-dev
EXTPATH = /code/neurdb-dev/dbengine/nr_kernel/nr_am
MODULE_big = nram
OBJS = ${EXTPATH}/src/nram.o


EXTENSION = nram
DATA = sql/nram--1.0.sql
PGFILEDESC = "nram - CCaaLF custom table access method"

REGRESS =

PG_CONFIG = ${NEURDBPATH}/psql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
