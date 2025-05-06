MODULE_big = ccam
OBJS = ccam.o

EXTENSION = ccam
DATA = ccam--1.0.sql
PGFILEDESC = "ccam - CCaaLF custom table access method"

REGRESS = 

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
