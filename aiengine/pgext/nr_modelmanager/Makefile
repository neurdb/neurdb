# Make file for the nr_model module
# It calls the CMakelists.txt file in the same directory

all:
	mkdir -p build
	cd build && cmake .. -DTorch_DIR:PATH=/home/libtorch/share/cmake/Torch	# path to libtorch in the docker container

install:
	cd build && make install
	cd ..
#	cp nr_model.control $(shell pg_config --sharedir)/extension
#	cp sql/nr_model--1.0.0.sql $(shell pg_config --sharedir)/extension
#	cp build/libnr_model.so $(shell pg_config --libdir)

clean:
	rm -rf build
