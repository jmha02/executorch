cmake_minimum_required(VERSION 3.19)
project(custom_cpu_runner)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
  flatccrt
)

set_target_properties(
  custom_cpu_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)

# QNN-enabled runner (reuses same main.cpp)
add_executable(custom_qnn_runner main.cpp)
target_include_directories(custom_qnn_runner PUBLIC
  ${EXECUTORCH_ROOT}/..
  ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)
target_link_libraries(custom_qnn_runner
  executorch_core
  extension_module
  extension_evalue_util
  extension_runner_util
  portable_ops_lib
  flatccrt
  qnn_backend
  qnn_executorch_backend
)
set_target_properties(
  custom_qnn_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
