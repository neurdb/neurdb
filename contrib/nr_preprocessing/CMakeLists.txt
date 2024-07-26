cmake_minimum_required(VERSION 3.21)
project(nr_preprocessing C CXX)

set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CMAKE_C_COMPILER /usr/bin/gcc-13)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PostgreSQL_ROOT /code/neurdb-dev/psql)
#set(PostgreSQL_ROOT /usr/local/pgsql)
set(PostgreSQL_LIBRARY_DIRS "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_INCLUDE_DIRS "${PostgreSQL_ROOT}/include")
set(PostgreSQL_LIBDIR "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_SHAREDIR "${PostgreSQL_ROOT}/share")

# PostgreSQL
find_package(PostgreSQL REQUIRED)
include_directories(${PostgreSQL_INCLUDE_DIRS})

# CURL
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# libwebsockets
find_library(LWS_LIBRARY NAMES websockets)

# socket.io
find_library(SIO_CLIENT NAMES sioclient)

FILE(GLOB_RECURSE SOURCES "src/*.c" "src/*.cpp")
add_library(nr_preprocessing SHARED ${SOURCES})
target_link_libraries(nr_preprocessing ${PostgreSQL_LIBRARIES} ${CURL_LIBRARIES} ${LWS_LIBRARY} ${SIO_CLIENT})

install(TARGETS nr_preprocessing DESTINATION ${PostgreSQL_LIBDIR})
install(FILES sql/nr_preprocessing--1.0.0.sql DESTINATION ${PostgreSQL_SHAREDIR}/extension)
