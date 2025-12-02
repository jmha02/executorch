cmake_minimum_required(VERSION 3.19)
project(custom_cpu_runner)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(executorch CONFIG REQUIRED)
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)

add_executable(custom_cpu_runner main.cpp)

target_include_directories(custom_cpu_runner PUBLIC
  ${EXECUTORCH_ROOT}/..
  ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

target_link_libraries(custom_cpu_runner
  executorch_core
  extension_module
  extension_evalue_util
  extension_runner_util
  portable_ops_lib
  etdump
  flatccrt
)

set_target_properties(
  custom_cpu_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
