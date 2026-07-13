cmake_minimum_required(VERSION 3.21)
project(nr_molqo C CXX)

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

include_directories(${PostgreSQL_INCLUDE_DIRS})

# Base dir
include_directories(${nr_molqo_SOURCE_DIR}/src)

# Add library - only compile nr_molqo.c for now
# HTTP client (http_client.c) is reserved for future implementation
set(SOURCES
    src/nr_molqo.c
    # src/http_client.c  # Uncomment when implementing real HTTP client
)
add_library(nr_molqo SHARED ${SOURCES})

# Optional: Add HTTP client dependencies when ready
# find_package(CURL REQUIRED)
# include_directories(${CURL_INCLUDE_DIRS})
# target_link_libraries(nr_molqo ${CURL_LIBRARIES})

# Install
install(TARGETS nr_molqo DESTINATION ${PostgreSQL_LIBDIR})
install(FILES sql/nr_molqo--1.0.sql DESTINATION ${PostgreSQL_SHAREDIR}/extension)
