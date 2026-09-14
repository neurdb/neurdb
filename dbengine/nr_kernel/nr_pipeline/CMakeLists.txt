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

# Discover PostgreSQL paths via pg_config
if(NOT DEFINED PG_CONFIG)
    find_program(PG_CONFIG pg_config REQUIRED)
endif()
execute_process(COMMAND ${PG_CONFIG} --includedir-server OUTPUT_VARIABLE PostgreSQL_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PG_CONFIG} --pkglibdir       OUTPUT_VARIABLE PostgreSQL_LIBDIR        OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PG_CONFIG} --sharedir        OUTPUT_VARIABLE PostgreSQL_SHAREDIR      OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PG_CONFIG} --libdir          OUTPUT_VARIABLE PostgreSQL_LIBRARY_DIRS  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PG_CONFIG} --libs            OUTPUT_VARIABLE PostgreSQL_LIBRARIES     OUTPUT_STRIP_TRAILING_WHITESPACE)

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
target_link_libraries(nr_pipeline ${CURL_LIBRARIES} ${LWS_LIBRARY} ${SIO_CLIENT} OpenSSL::SSL OpenSSL::Crypto)

# Install
install(TARGETS nr_pipeline DESTINATION ${PostgreSQL_LIBDIR})
install(FILES sql/nr_pipeline--2.0.0.sql DESTINATION ${PostgreSQL_SHAREDIR}/extension)
