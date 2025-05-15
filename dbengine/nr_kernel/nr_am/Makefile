MODULE_big = nram
OBJS = nram.o

NEURDBPATH=/code/neurdb-dev

EXTENSION = nram
DATA = nram--1.0.sql
PGFILEDESC = "nram - CCaaLF custom table access method"

REGRESS =

PG_CONFIG = ${NEURDBPATH}/psql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
