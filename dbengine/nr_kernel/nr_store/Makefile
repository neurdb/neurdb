# nr_store/Makefile

EXT_NAME = pg_neurstore
EXT_NAME_CORE = neurstore_core

PG_CONFIG ?= pg_config
PG_SHAREDIR  := $(shell $(PG_CONFIG) --sharedir)
PG_PKGLIBDIR := $(shell $(PG_CONFIG) --pkglibdir)

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
	cd $(BUILD_DIR) && $(CMAKE) .. -DPG_CONFIG=$(PG_CONFIG)

build: configure
	$(CMAKE) --build $(BUILD_DIR)

install: build
	cd $(BUILD_DIR) && $(CMAKE) --build .
	cp $(EXT_NAME).control $(PG_SHAREDIR)/extension
	cp sql/pg_neurstore--1.0.0.sql $(PG_SHAREDIR)/extension
	cp build/$(EXT_NAME).so $(PG_PKGLIBDIR)
	cp build/$(EXT_NAME).so $(PG_PKGLIBDIR)/postgresql
	cp build/$(EXT_NAME_CORE).so $(PG_PKGLIBDIR)
	cp build/$(EXT_NAME_CORE).so $(PG_PKGLIBDIR)/postgresql

clean:
	rm -rf $(BUILD_DIR)
