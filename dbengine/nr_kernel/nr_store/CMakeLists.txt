cmake_minimum_required(VERSION 3.25)
project(pg_neurstore C CXX)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

############   CMake options ############
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma")

############   CMake options (Debug) ############
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_C_FLAGS_DEBUG "-g -O0 -march=native -mavx2 -mfma")
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -march=native -mavx2 -mfma")

############ PG_ROOT option for custom PostgreSQL installation ############
option(PG_ROOT "Root installation directory of Postgres")
if (PG_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${PG_ROOT}")
endif ()

############ Find packages ############
# postgresql
find_package(PostgreSQL REQUIRED)
if (NOT PostgreSQL_FOUND)
    message(FATAL_ERROR "PostgreSQL not found")
endif ()
find_path(PostgreSQL_SERVER_INCLUDE_DIR
        NAMES postgres.h
        HINTS ${PostgreSQL_INCLUDE_DIRS}
        PATH_SUFFIXES server 16/server
)
if (NOT PostgreSQL_SERVER_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find PostgreSQL server include directory (postgres.h)")
endif ()

# eigen
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)

# onnx
set(ONNX_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnx")
find_library(ONNX_RUNTIME_LIBRARY NAMES onnxruntime HINTS ${ONNX_RUNTIME_DIR}/lib)
include_directories(${ONNX_RUNTIME_DIR}/include)

# protobuf
set(PROTOBUF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/protobuf/protobuf_install")
find_library(PROTOBUF_LIBRARIES NAMES protobuf HINTS ${PROTOBUF_DIR}/lib)
find_library(PROTOC_LIBRARIES NAMES protoc HINTS ${PROTOBUF_DIR}/lib)
include_directories(${PROTOBUF_DIR}/include)

# onnx generated protobuf
set(ONNX_PROTOBUF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnx/onnx_protobuf/build")
include_directories(${ONNX_PROTOBUF_DIR})

# hnswlib (header-only INTERFACE library)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/hnswlib)

# openmp
find_package(OpenMP REQUIRED)

# pqxx
find_package(PkgConfig REQUIRED)
pkg_check_modules(PQXX REQUIRED libpqxx)

# tokenizers-cpp
set(TOKENIZERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/tokenizers-cpp/build")
find_library(TOKENIZERS_CPP_LIB NAMES tokenizers_cpp HINTS ${TOKENIZERS_DIR} REQUIRED)
find_library(TOKENIZERS_C_LIB NAMES tokenizers_c HINTS ${TOKENIZERS_DIR} REQUIRED)
find_library(SENTENCEPIECE_LIB NAMES tokenizers_c HINTS ${TOKENIZERS_DIR} REQUIRED)
include_directories(${TOKENIZERS_DIR}/../include)

# openCV
find_package(OpenCV REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

# neurstore_core
file(GLOB ONNX_PROTO_SOURCE "${ONNX_PROTOBUF_DIR}/*.cc")
file(GLOB_RECURSE NEURSTORE_CORE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/neurstore/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/neurstore/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/neurstore/*.cpp
        ${ONNX_PROTO_SOURCE}
)
add_library(neurstore_core SHARED ${NEURSTORE_CORE})

target_include_directories(
        neurstore_core
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PostgreSQL_INCLUDE_DIRS}
        ${PostgreSQL_SERVER_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)
target_link_libraries(
        neurstore_core
        hnswlib
        ${PostgreSQL_LIBRARIES}
        OpenMP::OpenMP_CXX
        ${PROTOBUF_DIR}/lib/libprotobuf.a
        ${ONNX_RUNTIME_LIBRARY}
        ${TOKENIZERS_CPP_LIB}
        ${SENTENCEPIECE_LIB}
        ${OpenCV_LIBS}
)

set_target_properties(neurstore_core PROPERTIES
        INSTALL_RPATH "${ONNX_RUNTIME_DIR}/lib"
)

# microservice
file(GLOB_RECURSE NS_MICRO_SERVICE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microservice/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microservice/*.cpp
)

add_library(neurstore_microservice SHARED ${NS_MICRO_SERVICE})

target_include_directories(
        neurstore_microservice
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# pg_neurstore
FILE(GLOB_RECURSE NEURSTORE_PG
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pgext/*.c
)
add_library(pg_neurstore SHARED ${NEURSTORE_PG})
#set_target_properties(pg_neurstore PROPERTIES
#        OUTPUT_NAME "pg_neurstore"
#)

target_include_directories(
        pg_neurstore
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PostgreSQL_INCLUDE_DIRS}
        ${PostgreSQL_SERVER_INCLUDE_DIR}
)
target_link_libraries(
        pg_neurstore
        neurstore_core
        neurstore_microservice
        ${PostgreSQL_LIBRARIES}
)

set_target_properties(pg_neurstore PROPERTIES
        INSTALL_RPATH "$ORIGIN:${ONNX_RUNTIME_DIR}/lib"
)

############ Installation ############
execute_process(
        COMMAND pg_config --sharedir
        OUTPUT_VARIABLE PG_SHARE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PostgreSQL_EXTENSION_DIR "${PG_SHARE_DIR}/extension")

execute_process(
        COMMAND pg_config --pkglibdir
        OUTPUT_VARIABLE PG_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(TARGETS pg_neurstore neurstore_core
        LIBRARY DESTINATION ${PG_LIB_DIR})

install(FILES pg_neurstore.control sql/pg_neurstore--1.0.0.sql
        DESTINATION ${PostgreSQL_EXTENSION_DIR})

############ Example ############
file(GLOB EXAMPLES
        ${CMAKE_CURRENT_SOURCE_DIR}/example/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/example/*.cpp
)

foreach (EXAMPLE ${EXAMPLES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE})
    target_include_directories(
            ${EXAMPLE_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${PostgreSQL_INCLUDE_DIRS}
            ${PostgreSQL_SERVER_INCLUDE_DIR}
    )
    target_link_libraries(
            ${EXAMPLE_NAME}
            neurstore_core
            ${PostgreSQL_LIBRARIES}
            ${PROTOBUF_LIBRARIES}
            ${ONNX_RUNTIME_LIBRARY}
            ${TOKENIZERS_CPP_LIB}
            ${SENTENCEPIECE_LIB}
    )
endforeach ()
