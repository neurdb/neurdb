# Make file for the nr_preprocessing module
# It calls the CMakelists.txt file in the same directory

NEURDBPATH ?= /code/neurdb-dev

.PHONY: all cmake buildext install clean

all: cmake buildext

build:
	mkdir -p build
	cd build && cmake ..

buildext: build
	cd build; make -j; cd ..

install: buildext
	cd build; make install; cd ..
	cp nr_preprocessing.control $(NEURDBPATH)/psql/share/postgresql/extension
	cp sql/nr_preprocessing--1.0.0.sql $(NEURDBPATH)/psql/share/postgresql/extension
	# only works when in psql/lib/postgresql
	rm -f $(NEURDBPATH)/psql/lib/libnr_preprocessing.so
	rm -f $(NEURDBPATH)/psql/lib/postgresql/libnr_preprocessing.so
	cp build/libnr_preprocessing.so $(NEURDBPATH)/psql/lib
	cp build/libnr_preprocessing.so $(NEURDBPATH)/psql/lib/postgresql

clean:
	rm -rf build
