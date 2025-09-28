NEURDBPATH=/code/neurdb-dev
EXTPATH = /code/neurdb-dev/dbengine/nr_kernel/nr_am
ENGINEPATH = /code/neurdb-dev/dbengine/
MODULE_big = nram
OBJS = \
	${EXTPATH}/src/nram.o \
	${EXTPATH}/src/nram_access/kv.o \
	${EXTPATH}/src/nram_xact/xact.o \
	${EXTPATH}/src/nram_xact/action.o \
	${EXTPATH}/src/nram_storage/rocksengine.o \
	${EXTPATH}/src/nram_storage/rocks_service.o \
	${EXTPATH}/src/nram_storage/thread.o \
	${EXTPATH}/src/nram_storage/rocks_handler.o \
	${EXTPATH}/src/ipc/msg.o \
	${EXTPATH}/src/test/kv_test.o \
	${EXTPATH}/src/test/channel_test.o \
	${EXTPATH}/src/test/action_test.o \
	${EXTPATH}/src/nram_utils/config.o



EXTENSION = nram
DATA = sql/nram--1.0.sql
PGFILEDESC = "nram - CCaaLF custom table access method"

# Link to RocksDB C API
PG_CPPFLAGS += -I/usr/local/include -I./src/
SHLIB_LINK += -L/usr/local/lib -lrocksdb -Wl,-rpath,/usr/local/lib

REGRESS = clean init unit_tests table_scan index_scan xact_basic

ISOLATION = ssi_read_committed rc_read_committed \
			repeatable_read repeatable_read_2pl \
			lost_update lost_update_2pl

ISOLATION_OPTS = --temp-config $(top_srcdir)/src/test/regress/postgresql.conf
ISOLATIONDIR = $(srcdir)/isolation
ISOLATION_EXPECTED = $(ISOLATIONDIR)
ISOLATION_OUTPUT = $(ISOLATIONDIR)
ISOLATION_SQL = $(ISOLATIONDIR)

ROCKSDB_LIB_PATH = /usr/local/lib
ROCKSDB_INC_PATH = /usr/local/include


PG_CONFIG = ${NEURDBPATH}/psql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

.PHONY: debug test

debug:#
	$(MAKE) PG_CPPFLAGS="$(PG_CPPFLAGS) -DDEBUG" \
	        SHLIB_LINK="$(SHLIB_LINK)" \
	        CFLAGS="-g -O0 -Wall"

test:
	rm -rf ../logfile
	rm -rf "${NEURDBPATH}/psql/data/pg_rocksdb/"
	make clean
	make debug -j
	sudo make install
	bash ../scripts/restart_db.sh
	# psql -d neurdb -f ./sql/clean.sql
	# psql -d neurdb -f ./sql/init.sql
	# psql -d neurdb -f ./sql/unit_tests.sql
	make installcheck

unit-test:
	rm -rf ../logfile
	rm -rf "${NEURDBPATH}/psql/data/pg_rocksdb/"
	make clean
	make debug -j
	sudo make install
	bash ../scripts/restart_db.sh
	echo "Running unit tests..."
	psql -d neurdb -f ./sql/clean.sql
	psql -d neurdb -f ./sql/init.sql
	psql -d neurdb -f ./sql/unit_tests.sql


setup:
	rm -rf ../logfile
	rm -rf "${NEURDBPATH}/psql/data/pg_rocksdb/"
	make clean
	make debug -j
	sudo make install
	bash ../scripts/restart_db.sh
	psql -d neurdb -f ./sql/clean.sql
	psql -d neurdb -f ./sql/init.sql

