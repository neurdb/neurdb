# Make file for NeurDB MoLQO
# It calls the CMakeLists.txt file in the same directory

EXT_NAME = nr_molqo

PG_CONFIG ?= pg_config
PG_SHAREDIR  := $(shell $(PG_CONFIG) --sharedir)
PG_PKGLIBDIR := $(shell $(PG_CONFIG) --pkglibdir)

.PHONY: all cmake buildext install clean debug builddbg buildextdbg

all: build buildext

build:
	mkdir -p build
	cd build && cmake -DPG_CONFIG=$(PG_CONFIG) -DCMAKE_BUILD_TYPE=Release ..

debug: builddbg buildextdbg

builddbg:
	mkdir -p build
	cd build && cmake -DPG_CONFIG=$(PG_CONFIG) -DCMAKE_BUILD_TYPE=Debug ..

buildextdbg: buildext
buildext: build
	$(MAKE) -j -C build

install: buildext
	$(MAKE) -C build install
	cp $(EXT_NAME).control $(PG_SHAREDIR)/extension

clean:
	rm -rf build
