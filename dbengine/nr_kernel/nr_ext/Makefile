# Make file for NeurDB kernel
# It calls the CMakelists.txt file in the same directory

EXT_NAME = nr_ext

NEURDBPATH ?= /code/neurdb-dev

.PHONY: all cmake buildext install clean

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
	cp $(EXT_NAME).control $(NEURDBPATH)/psql/share/postgresql/extension
	# only works when in psql/lib/postgresql
	rm -f $(NEURDBPATH)/psql/lib/$(EXT_NAME).so
	rm -f $(NEURDBPATH)/psql/lib/postgresql/$(EXT_NAME).so
	cp build/$(EXT_NAME).so $(NEURDBPATH)/psql/lib
	cp build/$(EXT_NAME).so $(NEURDBPATH)/psql/lib/postgresql

clean:
	rm -rf build
