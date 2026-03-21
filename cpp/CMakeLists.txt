cmake_minimum_required(VERSION 3.16)
project(cylon_armada VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# Find Cylon (installed headers + library)
# ---------------------------------------------------------------------------

if (NOT CYLON_PREFIX)
    message(FATAL_ERROR "CYLON_PREFIX must be set to the Cylon install directory")
endif ()

include_directories(${CYLON_PREFIX}/include)
link_directories(${CYLON_PREFIX}/lib)

# Find Arrow (same as Cylon's build)
find_package(Arrow REQUIRED)
message(STATUS "Arrow version: ${Arrow_VERSION}")

# ---------------------------------------------------------------------------
# cylon_armada shared library
# ---------------------------------------------------------------------------

set(ARMADA_SOURCES
        src/context/context_table.hpp
        src/context/context_table.cpp
        src/context/context_serialize.hpp
        src/context/context_serialize.cpp)

add_library(cylon_armada SHARED ${ARMADA_SOURCES})

target_include_directories(cylon_armada PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CYLON_PREFIX}/include)

target_link_libraries(cylon_armada cylon ${ARROW_SHARED_LIB})

# Redis persistence (optional)
if (CYLON_USE_REDIS)
    message("cylon-armada Redis Enabled")
    target_compile_definitions(cylon_armada PRIVATE BUILD_CYLON_REDIS)

    find_path(HIREDIS_HEADER hiredis HINTS ${REDIS_INSTALL_PREFIX}/include)
    target_include_directories(cylon_armada PUBLIC ${HIREDIS_HEADER})

    find_library(HIREDIS_LIB hiredis HINTS ${REDIS_INSTALL_PREFIX}/lib)
    find_library(REDIS_PLUS_PLUS_LIB redis++ HINTS ${REDIS_INSTALL_PREFIX}/lib64 ${REDIS_INSTALL_PREFIX}/lib)
    target_link_libraries(cylon_armada ${HIREDIS_LIB} ${REDIS_PLUS_PLUS_LIB})
endif ()

# S3 persistence (optional, uses AWS SDK from FMI build)
if (CYLON_FMI)
    message("cylon-armada S3 Enabled")
    find_package(AWSSDK COMPONENTS s3 REQUIRED)
    target_compile_definitions(cylon_armada PRIVATE BUILD_CYLON_FMI)
    target_link_libraries(cylon_armada ${AWSSDK_LINK_LIBRARIES})
endif ()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install(TARGETS cylon_armada LIBRARY DESTINATION lib)
install(FILES
        src/context/context_table.hpp
        src/context/context_serialize.hpp
        DESTINATION include/cylon_armada/context)