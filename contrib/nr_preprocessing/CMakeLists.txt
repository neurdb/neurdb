cmake_minimum_required(VERSION 3.21)
project(nr_preprocessing C)

set(CMAKE_C_STANDARD 17)

set(PostgreSQL_ROOT /code/neurdb-dev/psql)
#set(PostgreSQL_ROOT /usr/local/pgsql)
set(PostgreSQL_LIBRARY_DIRS "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_INCLUDE_DIRS "${PostgreSQL_ROOT}/include")
set(PostgreSQL_LIBDIR "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_SHAREDIR "${PostgreSQL_ROOT}/share")

find_package(PostgreSQL REQUIRED)
include_directories(${PostgreSQL_INCLUDE_DIRS})

FILE(GLOB_RECURSE SOURCES "src/*.c")
add_library(nr_preprocessing SHARED ${SOURCES})
target_link_libraries(nr_preprocessing ${PostgreSQL_LIBRARIES})

install(TARGETS nr_preprocessing DESTINATION ${PostgreSQL_LIBDIR})
install(FILES sql/nr_preprocessing--1.0.0.sql DESTINATION ${PostgreSQL_SHAREDIR}/extension)
