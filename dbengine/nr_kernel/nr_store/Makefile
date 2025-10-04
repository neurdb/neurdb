# nr_store/Makefile

EXT_NAME = pg_neurstore
EXT_NAME_CORE = neurstore_core

NEURDBPATH ?= /code/neurdb-dev

BUILD_DIR ?= build
CMAKE     ?= cmake
PREFIX    ?= /usr/local

.PHONY: all cargo external configure build install clean

all: build

cargo:
	@which cargo >/dev/null 2>&1 || \
	( echo "[SETUP] Installing cargo via rustup..." && \
	  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y )

external: cargo
	bash script/build_external.sh

configure: external
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_INSTALL_PREFIX=$(PREFIX) ..

build: configure
	$(CMAKE) --build $(BUILD_DIR)

install: build
	cd $(BUILD_DIR) && $(CMAKE) --build . --target install
	cp $(EXT_NAME).control $(NEURDBPATH)/psql/share/postgresql/extension
	cp sql/pg_neurstore--1.0.0.sql $(NEURDBPATH)/psql/share/postgresql/extension
	cp build/lib$(EXT_NAME).so $(NEURDBPATH)/psql/lib
	cp build/lib$(EXT_NAME).so $(NEURDBPATH)/psql/lib/postgresql
	cp build/lib$(EXT_NAME_CORE).so $(NEURDBPATH)/psql/lib
	cp build/lib$(EXT_NAME_CORE).so $(NEURDBPATH)/psql/lib/postgresql

clean:
	rm -rf $(BUILD_DIR)
