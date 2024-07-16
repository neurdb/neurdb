cmake_minimum_required(VERSION 3.21)
project(nr_model C CXX)

#set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
#set(CMAKE_C_COMPILER /usr/bin/gcc-13)

# Set make standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set PostgreSQL paths
set(PostgreSQL_ROOT /code/neurdb-dev/psql)
#set(PostgreSQL_ROOT /usr/local/pgsql)
set(PostgreSQL_LIBRARY_DIRS "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_INCLUDE_DIRS "${PostgreSQL_ROOT}/include")
set(PostgreSQL_LIBDIR "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_SHAREDIR "${PostgreSQL_ROOT}/share")


# Check PostgreSQL and Torch
find_package(PostgreSQL REQUIRED)
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS} ${PostgreSQL_INCLUDE_DIRS})

# Add RPATH for Torch libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${TORCH_LIBRARY_DIRS}")

FILE(GLOB_RECURSE SOURCES src/*.c src/*.cpp src/*.h src/*.hpp)
add_library(nr_model SHARED ${SOURCES})
target_link_libraries(nr_model ${TORCH_LIBRARIES} ${PostgreSQL_LIBRARIES})

# Install
install(TARGETS nr_model
        LIBRARY DESTINATION ${PostgreSQL_LIBDIR})
install(FILES nr_model.control sql/nr_model--1.0.0.sql
        DESTINATION ${PostgreSQL_SHAREDIR}/extension)
