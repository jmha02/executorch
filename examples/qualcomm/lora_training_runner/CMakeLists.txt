# LoRA Training Runner for On-Device Training

set(_lora_training_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/lora_training_runner.cpp
)

# Build lora training runner
add_executable(lora_training_runner ${_lora_training_runner__srcs})

target_include_directories(
  lora_training_runner PUBLIC
  ${EXECUTORCH_SOURCE_DIR}/..
  ${EXECUTORCH_SOURCE_DIR}/backends/qualcomm
  ${EXECUTORCH_SOURCE_DIR}/extension/data_loader
)

target_link_libraries(
  lora_training_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_evalue_util
  extension_runner_util
  full_portable_ops_lib  # ADD THIS for portable operators
  etdump
  flatccrt
)

set_target_properties(
  lora_training_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
