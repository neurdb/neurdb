cmake_minimum_required(VERSION 3.21)
project(nram C CXX)

set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Set make standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "-Werror=implicit-function-declaration")

# Set PostgreSQL paths
set(PostgreSQL_ROOT /code/neurdb-dev/psql)
set(PostgreSQL_LIBRARY_DIRS "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_INCLUDE_DIRS "${PostgreSQL_ROOT}/include")
set(PostgreSQL_LIBDIR "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_SHAREDIR "${PostgreSQL_ROOT}/share")

# Check PostgreSQL
find_package(PostgreSQL REQUIRED)
include_directories(
    ${PostgreSQL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add library
FILE(GLOB_RECURSE SOURCES src/*.c src/*.cpp src/*.h src/*.hpp)
add_library(nram SHARED ${SOURCES})
target_link_libraries(nram ${PostgreSQL_LIBRARIES})

# Install
install(
    TARGETS nram
    LIBRARY DESTINATION ${PostgreSQL_LIBDIR}
)
