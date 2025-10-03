# nr_store/Makefile

EXT_NAME = pg_neurstore
EXT_NAME_CORE = neurstore_core

NEURDBPATH ?= /code/neurdb-dev

BUILD_DIR ?= build
CMAKE     ?= cmake
PREFIX    ?= /usr/local

.PHONY: all external configure build install clean

all: build

external:
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
