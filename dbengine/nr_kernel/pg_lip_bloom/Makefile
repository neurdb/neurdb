EXTENSION = pg_lip_bloom
MODULE_big = pg_lip_bloom
OBJS = pg_lip_bloom.o bloom.o murmur2/MurmurHash2.o
DATA = pg_lip_bloom--1.0.sql

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

SHLIB_LINK += -lm
