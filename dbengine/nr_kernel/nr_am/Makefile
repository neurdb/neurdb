NEURDBPATH=/code/neurdb-dev
EXTPATH = /code/neurdb-dev/dbengine/nr_kernel/nr_am
ENGINEPATH = /code/neurdb-dev/dbengine/
MODULE_big = nram
OBJS = \
	${EXTPATH}/src/nram.o \
	${EXTPATH}/src/nram_access/kv.o \
	${EXTPATH}/src/nram_xact/xact.o \
	${EXTPATH}/src/nram_storage/rocksengine.o \
	${EXTPATH}/src/nram_storage/rocks_service.o \
	${EXTPATH}/src/nram_storage/thread.o \
	${EXTPATH}/src/nram_storage/rocks_handler.o \
	${EXTPATH}/src/ipc/msg.o \
	${EXTPATH}/src/test/kv_test.o \
	${EXTPATH}/src/test/channel_test.o \
	${EXTPATH}/src/nram_utils/config.o



EXTENSION = nram
DATA = sql/nram--1.0.sql
PGFILEDESC = "nram - CCaaLF custom table access method"

# Link to RocksDB C API
PG_CPPFLAGS += -I/usr/local/include -I./src/
SHLIB_LINK += -L/usr/local/lib -lrocksdb -Wl,-rpath,/usr/local/lib

REGRESS = clean init unit_tests table_scan index_scan xact_basic multi_insert

# ISOLATION = read_committed repeatable_read
# ISOLATION = read_committed
ISOLATION_OPTS = --temp-config $(top_srcdir)/src/test/regress/postgresql.conf
ISOLATIONDIR = $(srcdir)/isolation
ISOLATION_EXPECTED = $(ISOLATIONDIR)
ISOLATION_OUTPUT = $(ISOLATIONDIR)
ISOLATION_SQL = $(ISOLATIONDIR)

ROCKSDB_SERVER_SRC = ${EXTPATH}/server/server.c
ROCKSDB_SERVER_BIN = ${EXTPATH}/server/server
ROCKSDB_LIB_PATH = /usr/local/lib
ROCKSDB_INC_PATH = /usr/local/include


PG_CONFIG = ${NEURDBPATH}/psql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

.PHONY: debug test # rocks_server clean_rocks_server

debug:
	$(MAKE) PG_CPPFLAGS="$(PG_CPPFLAGS) -DDEBUG" \
	        SHLIB_LINK="$(SHLIB_LINK)" \
	        CFLAGS="-g -O0 -Wall"

# rocks_server: $(ROCKSDB_SERVER_BIN)

# $(ROCKSDB_SERVER_BIN): $(ROCKSDB_SERVER_SRC)
# 	gcc -o $@ $< -I$(ROCKSDB_INC_PATH) -L$(ROCKSDB_LIB_PATH) -lrocksdb -Wl,-rpath,$(ROCKSDB_LIB_PATH)

# clean_rocks_server:
# 	rm -rf "${NEURDBPATH}/psql/data/pg_rocksdb/"
# 	rm -rf $(ROCKSDB_SERVER_BIN)
#  clean_rocks_server rocks_server

test:
	rm -rf ../logfile
	rm -rf "${NEURDBPATH}/psql/data/pg_rocksdb/"
	$(ROCKSDB_SERVER_BIN) & sleep 1  # Give server time to start
	make clean
	make debug -j
	sudo make install
	bash ../scripts/restart_db.sh
	# psql -d neurdb -f ./sql/clean.sql
	# psql -d neurdb -f ./sql/init.sql
	# psql -d neurdb -f ./sql/unit_tests.sql
	make installcheck
