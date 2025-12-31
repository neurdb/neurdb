cmake_minimum_required(VERSION 3.21)
project(nr_pipeline C CXX)

set(CMAKE_SHARED_LIBRARY_PREFIX "")

#set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
#set(CMAKE_C_COMPILER /usr/bin/gcc-13)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PostgreSQL_ROOT /code/neurdb-dev/build/psql)
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

# Base dir
include_directories(${nr_pipeline_SOURCE_DIR}/src)

# openssl
find_package(OpenSSL REQUIRED)

FILE(GLOB_RECURSE SOURCES "src/*.c" "src/*.cpp")
LIST(FILTER SOURCES EXCLUDE REGEX ".*src/interface.[ch]$") # use interface2.xxx as entry point
LIST(FILTER SOURCES EXCLUDE REGEX ".*src/utils/network/http.c$")
add_library(nr_pipeline SHARED ${SOURCES})
target_link_libraries(nr_pipeline ${PostgreSQL_LIBRARIES} ${CURL_LIBRARIES} ${LWS_LIBRARY} ${SIO_CLIENT} OpenSSL::SSL OpenSSL::Crypto)

# Install
install(TARGETS nr_pipeline DESTINATION ${PostgreSQL_LIBDIR})
install(FILES sql/nr_pipeline--2.0.0.sql DESTINATION ${PostgreSQL_SHAREDIR}/extension)
