# Make file for the pg_model module
# It calls the CMakelists.txt file in the same directory

all:
	mkdir -p build
	cd build && cmake .. -DTorch_DIR:PATH=/home/siqi/Desktop/thrid-party/libtorch_cpu/share/cmake/Torch
	cd build && make

install:
	cd build && make install
	cp pg_model.control $(shell pg_config --sharedir)/extension
	cp sql/pg_model--1.0.0.sql $(shell pg_config --sharedir)/extension
	cp build/libpg_model.so $(shell pg_config --libdir)

clean:
	rm -rf build