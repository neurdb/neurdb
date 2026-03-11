.PHONY: help deps build install install-dbengine pg_hint_plan kernel_ext install-api install-aiengine initdb start-db start-ai start stop-db stop-ai stop clean distclean

# Default variables as described in INSTALL.md
NEURDBPATH ?= $(CURDIR)
NR_BUILD_PATH ?= $(NEURDBPATH)/build
NR_PSQL_PATH ?= $(NR_BUILD_PATH)/psql
NR_DBDATA_PATH ?= $(NR_BUILD_PATH)/data

# Multiarch libdir, defaults to x86_64-linux-gnu if dpkg-architecture is unavailable
MULTIARCH_LIBDIR ?= /usr/lib/$(shell dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)
export PKG_CONFIG_PATH := $(MULTIARCH_LIBDIR)/pkgconfig:$(PKG_CONFIG_PATH)
export LD_LIBRARY_PATH := $(MULTIARCH_LIBDIR):$(LD_LIBRARY_PATH)

# libclang path
export LIBCLANG_PATH ?= $(shell llvm-config --libdir 2>/dev/null)

NR_DBENGINE_PATH ?= $(NEURDBPATH)/dbengine
NR_KERNEL_PATH ?= $(NR_DBENGINE_PATH)/nr_kernel
NR_AIENGINE_PATH ?= $(NEURDBPATH)/aiengine
NR_API_PATH ?= $(NEURDBPATH)/api

AI_ENGINE_MODE ?= cpu

help:
	@echo "NeurDB Native Build Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make deps          - Install system dependencies for Linux native build"
	@echo "  make build         - Build the DB engine (PostgreSQL)"
	@echo "  make install       - Install DB engine, extensions, API, and AI engine"
	@echo "                       (Use AI_ENGINE_MODE=gpu for GPU support)"
	@echo "  make start         - Initialize and start the database & AI engine"
	@echo "  make stop          - Stop the database and AI engine"
	@echo "  make clean         - Clean build artifacts (keeps data and psql binaries)"
	@echo "  make distclean     - Remove the entire build directory (including data)"

deps:
	@echo "========================================"
	@echo "==> Running target: deps"
	@echo "========================================"
	sudo apt-get update
	sudo apt-get install -y \
		python3-dev python3-pip python-is-python3 \
		build-essential gcc make cmake pkg-config \
		clang flex bison \
		libreadline-dev zlib1g-dev libicu-dev \
		libssl-dev libclang-dev llvm-dev \
		libcurl4-openssl-dev libwebsockets-dev libcjson-dev \
		librocksdb-dev libpqxx-dev libopencv-dev \
		curl git locales
	sudo locale-gen en_US.UTF-8
	sudo update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

build:
	@echo "========================================"
	@echo "==> Running target: build"
	@echo "========================================"
	mkdir -p $(NR_BUILD_PATH)/dbengine $(NR_PSQL_PATH)
	cd $(NR_BUILD_PATH)/dbengine && \
		$(NR_DBENGINE_PATH)/configure --prefix=$(NR_PSQL_PATH) --enable-debug && \
		$(MAKE) -j

install-dbengine: build
	@echo "========================================"
	@echo "==> Running target: install-dbengine"
	@echo "========================================"
	cd $(NR_BUILD_PATH)/dbengine && $(MAKE) install

pg_hint_plan: install-dbengine
	@echo "========================================"
	@echo "==> Running target: pg_hint_plan"
	@echo "========================================"
	mkdir -p $(NR_BUILD_PATH)/contrib
	cd $(NR_BUILD_PATH)/contrib && \
		if [ ! -d "pg_hint_plan" ]; then \
			git clone https://github.com/ossc-db/pg_hint_plan.git; \
		fi && \
		cd pg_hint_plan && git checkout PG16 && \
		$(MAKE) PG_CONFIG=$(NR_PSQL_PATH)/bin/pg_config clean || true && \
		$(MAKE) PG_CONFIG=$(NR_PSQL_PATH)/bin/pg_config && \
		$(MAKE) PG_CONFIG=$(NR_PSQL_PATH)/bin/pg_config install

kernel_ext: install-dbengine
	@echo "========================================"
	@echo "==> Running target: kernel_ext"
	@echo "========================================"
	cd $(NR_KERNEL_PATH) && \
		PG_CONFIG=$(NR_PSQL_PATH)/bin/pg_config $(MAKE) clean || true && \
		PG_CONFIG=$(NR_PSQL_PATH)/bin/pg_config $(MAKE) && \
		PG_CONFIG=$(NR_PSQL_PATH)/bin/pg_config $(MAKE) install

install-api:
	@echo "========================================"
	@echo "==> Running target: install-api"
	@echo "========================================"
	mkdir -p $(NR_BUILD_PATH)/api/python
	cp -r $(NR_API_PATH)/python/* $(NR_BUILD_PATH)/api/python/
	cd $(NR_BUILD_PATH)/api/python && touch setup.cfg && pip install -e . && rm setup.cfg

install-aiengine:
	@echo "========================================"
	@echo "==> Running target: install-aiengine"
	@echo "========================================"
ifeq ($(AI_ENGINE_MODE),cpu)
	pip install -r $(NR_AIENGINE_PATH)/runtime/requirements.cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
else
	pip install -r $(NR_AIENGINE_PATH)/runtime/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
endif

install: install-dbengine pg_hint_plan kernel_ext install-api install-aiengine
	@echo "========================================"
	@echo "==> Finished target: install"
	@echo "========================================"

initdb: install-dbengine
	@echo "========================================"
	@echo "==> Running target: initdb"
	@echo "========================================"
	@if [ ! -d "$(NR_DBDATA_PATH)" ]; then \
		mkdir -p $(NR_DBDATA_PATH); \
		$(NR_PSQL_PATH)/bin/initdb -D $(NR_DBDATA_PATH); \
	else \
		chmod 0750 $(NR_DBDATA_PATH); \
	fi

start-db: initdb
	@echo "========================================"
	@echo "==> Running target: start-db"
	@echo "========================================"
	$(NR_PSQL_PATH)/bin/pg_ctl -D $(NR_DBDATA_PATH) -l $(NR_BUILD_PATH)/logfile start || true
	@echo "Waiting for PostgreSQL to start..."
	@until $(NR_PSQL_PATH)/bin/psql -h localhost -p 5432 -U $(USER) -c '\q' 2>/dev/null; do \
		echo 'NeurDB is unavailable - sleeping'; \
		sleep 1; \
	done
	$(NR_PSQL_PATH)/bin/createdb -h localhost -p 5432 neurdb 2>/dev/null || true
	@echo "Configuring shared_preload_libraries..."
	sed -i '/^#*shared_preload_libraries/d' $(NR_DBDATA_PATH)/postgresql.conf
	echo "shared_preload_libraries = 'pg_hint_plan, nr_molqo, nr_ext, nram, pg_neurstore'" >> $(NR_DBDATA_PATH)/postgresql.conf
	$(NR_PSQL_PATH)/bin/pg_ctl -D $(NR_DBDATA_PATH) -l $(NR_BUILD_PATH)/logfile restart
	@echo "Waiting for PostgreSQL to restart..."
	@until $(NR_PSQL_PATH)/bin/psql -h localhost -p 5432 -U $(USER) -c '\q' 2>/dev/null; do \
		echo 'NeurDB is unavailable - sleeping'; \
		sleep 1; \
	done
	@echo "Creating nr_pipeline extension..."
	$(NR_PSQL_PATH)/bin/psql -h localhost -p 5432 -U $(USER) -d neurdb -c 'CREATE EXTENSION IF NOT EXISTS nr_pipeline;'

start-ai:
	@echo "========================================"
	@echo "==> Running target: start-ai"
	@echo "========================================"
	cd $(NR_AIENGINE_PATH)/runtime && NR_LOG_LEVEL=INFO nohup python server.py > $(NR_BUILD_PATH)/ai_engine.log 2>&1 &
	@echo -n 'Waiting for AI engine to start '
	@until curl --output /dev/null --silent --head --fail http://127.0.0.1:8090/; do \
		printf '.'; \
		sleep 1; \
	done
	@echo ' OK'

start: start-db start-ai
	@echo "========================================"
	@echo "==> Finished target: start"
	@echo "========================================"

stop-db:
	@echo "========================================"
	@echo "==> Running target: stop-db"
	@echo "========================================"
	$(NR_PSQL_PATH)/bin/pg_ctl -D $(NR_DBDATA_PATH) stop || true

stop-ai:
	@echo "========================================"
	@echo "==> Running target: stop-ai"
	@echo "========================================"
	pkill -f "python server.py" || true

stop: stop-db stop-ai
	@echo "========================================"
	@echo "==> Finished target: stop"
	@echo "========================================"

clean:
	@echo "========================================"
	@echo "==> Running target: clean"
	@echo "========================================"
	rm -rf $(NR_BUILD_PATH)/dbengine $(NR_BUILD_PATH)/contrib $(NR_BUILD_PATH)/api

distclean:
	@echo "========================================"
	@echo "==> Running target: distclean"
	@echo "========================================"
	rm -rf $(NR_BUILD_PATH)
