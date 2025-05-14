MODULE_big = ccam
OBJS = ccam.o

NEURDBPATH=/code/neurdb-dev

EXTENSION = ccam
DATA = ccam--1.0.sql
PGFILEDESC = "ccam - CCaaLF custom table access method"

REGRESS =

PG_CONFIG = ${NEURDBPATH}/psql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
