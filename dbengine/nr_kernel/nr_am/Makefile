NEURDBPATH=/code/neurdb-dev
EXTPATH = /code/neurdb-dev/dbengine/nr_kernel/nr_am
ENGINEPATH = /code/neurdb-dev/dbengine/
MODULE_big = nram
OBJS = \
	${EXTPATH}/src/nram.o \
	${EXTPATH}/src/nram_access/kv.o \
	${EXTPATH}/src/nram_xact/xact.o \
	${EXTPATH}/src/nram_storage/rocksengine.o \
	${EXTPATH}/src/ipc/mq.o \
	${EXTPATH}/src/ipc/msg.o \
	${EXTPATH}/src/test/kv_test.o



EXTENSION = nram
DATA = sql/nram--1.0.sql
PGFILEDESC = "nram - CCaaLF custom table access method"

# Link to RocksDB C API
PG_CPPFLAGS += -I/usr/local/include -I./src/
SHLIB_LINK += -L/usr/local/lib -lrocksdb -Wl,-rpath,/usr/local/lib

# REGRESS = clean init unit_tests table_scan xact_basic index_scan
REGRESS = clean init unit_tests

# ISOLATION = read_committed repeatable_read
# ISOLATION = read_committed
ISOLATION_OPTS = --temp-config $(top_srcdir)/src/test/regress/postgresql.conf
ISOLATIONDIR = $(srcdir)/isolation
ISOLATION_EXPECTED = $(ISOLATIONDIR)
ISOLATION_OUTPUT = $(ISOLATIONDIR)
ISOLATION_SQL = $(ISOLATIONDIR)


PG_CONFIG = ${NEURDBPATH}/psql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

.PHONY: debug test

debug:
	$(MAKE) -f Makefile PG_CPPFLAGS="$(PG_CPPFLAGS) -DDEBUG" SHLIB_LINK="$(SHLIB_LINK)" CFLAGS="-g -O0 -Wall"

test:
	rm -rf "${NEURDBPATH}/psql/data/pg_rocksdb/"
	rm -rf ../logfile
	make clean
	make -j
	sudo make install
	bash ../scripts/restart_db.sh
	psql -d neurdb -f ./sql/clean.sql
	psql -d neurdb -f ./sql/init.sql
	psql -d neurdb -f ./sql/unit_tests.sql
	# psql -d neurdb -f ./sql/table_scan.sql
	# make installcheck
