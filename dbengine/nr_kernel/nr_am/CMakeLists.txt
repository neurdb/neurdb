cmake_minimum_required(VERSION 3.21)
project(nram C CXX)

set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Set make standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "-Werror=implicit-function-declaration")

# Discover PostgreSQL paths via pg_config
if(NOT DEFINED PG_CONFIG)
    find_program(PG_CONFIG pg_config REQUIRED)
endif()
execute_process(COMMAND ${PG_CONFIG} --includedir-server OUTPUT_VARIABLE PostgreSQL_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PG_CONFIG} --pkglibdir       OUTPUT_VARIABLE PostgreSQL_LIBDIR        OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PG_CONFIG} --sharedir        OUTPUT_VARIABLE PostgreSQL_SHAREDIR      OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PG_CONFIG} --libdir          OUTPUT_VARIABLE PostgreSQL_LIBRARY_DIRS  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PG_CONFIG} --libs            OUTPUT_VARIABLE PostgreSQL_LIBRARIES     OUTPUT_STRIP_TRAILING_WHITESPACE)

include_directories(
    ${PostgreSQL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add library
FILE(GLOB_RECURSE SOURCES src/*.c src/*.cpp src/*.h src/*.hpp)
add_library(nram SHARED ${SOURCES})

# Install
install(
    TARGETS nram
    LIBRARY DESTINATION ${PostgreSQL_LIBDIR}
)
