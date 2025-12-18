# Make file for NeurDB MoLQO
# It calls the CMakeLists.txt file in the same directory

EXT_NAME = nr_molqo

NEURDBPATH ?= /code/neurdb-dev

.PHONY: all cmake buildext install clean debug builddbg buildextdbg

all: build buildext

build:
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=Release ..

debug: builddbg buildextdbg

builddbg:
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=Debug ..

buildextdbg: buildext
buildext:
	cd build; make -j; cd ..

install: buildext
	cd build; make install; cd ..
	cp $(EXT_NAME).control $(NEURDBPATH)/build/psql/share/postgresql/extension
	cp sql/nr_molqo--1.0.sql $(NEURDBPATH)/build/psql/share/postgresql/extension
	# only works when in psql/lib/postgresql
	rm -f $(NEURDBPATH)/build/psql/lib/$(EXT_NAME).so
	rm -f $(NEURDBPATH)/build/psql/lib/postgresql/$(EXT_NAME).so
	cp build/$(EXT_NAME).so $(NEURDBPATH)/build/psql/lib
	cp build/$(EXT_NAME).so $(NEURDBPATH)/build/psql/lib/postgresql

clean:
	rm -rf build
