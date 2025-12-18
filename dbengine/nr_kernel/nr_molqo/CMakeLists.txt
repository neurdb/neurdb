cmake_minimum_required(VERSION 3.21)
project(nr_molqo C CXX)

set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Set make standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "-Werror=implicit-function-declaration")

# Set PostgreSQL paths
set(PostgreSQL_ROOT /code/neurdb-dev/build/psql)
set(PostgreSQL_LIBRARY_DIRS "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_INCLUDE_DIRS "${PostgreSQL_ROOT}/include")
set(PostgreSQL_LIBDIR "${PostgreSQL_ROOT}/lib")
set(PostgreSQL_SHAREDIR "${PostgreSQL_ROOT}/share")

# Check PostgreSQL
find_package(PostgreSQL REQUIRED)
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
target_link_libraries(nr_molqo ${PostgreSQL_LIBRARIES})

# Optional: Add HTTP client dependencies when ready
# find_package(CURL REQUIRED)
# include_directories(${CURL_INCLUDE_DIRS})
# target_link_libraries(nr_molqo ${CURL_LIBRARIES})

# Install
install(TARGETS nr_molqo DESTINATION ${PostgreSQL_LIBDIR})
install(FILES sql/nr_molqo--1.0.sql DESTINATION ${PostgreSQL_SHAREDIR}/extension)
