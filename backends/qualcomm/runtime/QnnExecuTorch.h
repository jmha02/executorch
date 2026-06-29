/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#define QNN_BACKEND "QnnBackend"
#define QNN_RUNTIME_LOG_LEVEL "qnn_runtime_log_level"
#define QNN_RUNTIME_HTP_PERFORMANCE_MODE "qnn_runtime_htp_performance_mode"
#define QNN_RUNTIME_PROFILE_LEVEL "qnn_runtime_profile_level"
#define QNN_RUNTIME_LPAI_FPS "qnn_runtime_lpai_fps"
#define QNN_RUNTIME_LPAI_FTRT_RATIO "qnn_runtime_lpai_ftrt_ratio"
#define QNN_RUNTIME_LPAI_CLIENT_PERF_TYPE "qnn_runtime_lpai_client_perf_type"
#define QNN_RUNTIME_LPAI_AFFINITY "qnn_runtime_lpai_affinity"
#define QNN_RUNTIME_LPAI_CORE_SELECTION "qnn_runtime_lpai_core_selection"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// This could be:
// 1. qnn_context_binary
// 2. QnnContextCustomProtocol
// To check if it is custom protocol, users can deserialize the binary using
// QnnCustomProtocol and check the status
typedef struct {
  /// qnn_context_binary_blob
  void* buffer;
  /// number of bytes of buffer
  uint64_t nbytes;
} QnnExecuTorchContextBinary;

// clang-format off
#define QNN_EXECUTORCH_CONTEXT_BINARY    \
  {                                      \
    nullptr,        /*buffer*/           \
    0,              /*nbytes*/           \
  }
// clang-format on

/// Allocate memory in different way, check qnn document for more details.
enum QnnMemDescriptor { kIon, kCustom };

struct CustomMemTensorInfo {
  void* custom_mem;
  void* tensor_addr;
  size_t pos;
  size_t tensor_bytes;
  uint32_t* shape;
  uint32_t rank;
  executorch::aten::ScalarType dtype;
};

/// Allocate specific tensors (usually graph inputs and outputs) on shared
/// memory. Users are responsible to allocate "enough" tensor bytes, and set
/// alignment as MemoryAllocator::kDefaultAlignment.
/// See runtime/core/memory_allocator.h. The function returns a valid pointer
/// if allocation is successful.
__attribute__((__visibility__("default"))) void* QnnExecuTorchAllocCustomMem(
    size_t bytes,
    size_t alignment);

/// Add tensor to custom memory with custom type descriptor. Create memory
/// handle to tensor wrapper during execution
__attribute__((__visibility__("default"))) void
QnnExecuTorchAddCustomMemTensorAddr(void* tensor_addr, void* custom_mem);

/// Free the allocated shared memory.
__attribute__((__visibility__("default"))) void QnnExecuTorchFreeCustomMem(
    void* buffer_ptr);

typedef struct {
  uint64_t qnn_backend_execute_count;
  uint64_t qnn_register_mem_check_count;
  uint64_t qnn_register_mem_fallback_count;
  uint64_t qnn_register_mem_custom_base_hit_count;
  uint64_t qnn_register_mem_custom_base_miss_count;
  uint64_t qnn_register_ion_attempt_count;
  uint64_t qnn_register_custom_mem_entry_count;
  uint64_t qnn_shared_buffer_current_hit_count;
  uint64_t qnn_shared_buffer_preregistered_hit_count;
  uint64_t qnn_mem_register_count;
  uint64_t qnn_ion_mem_register_count;
  uint64_t qnn_custom_mem_register_count;
  uint64_t qnn_set_mem_handle_count;
  uint64_t qnn_fill_data_buffer_count;
  uint64_t qnn_fill_data_buffer_bytes;
  uint64_t qnn_backend_execute_us;
  uint64_t qnn_prepare_inputs_us;
  uint64_t qnn_prepare_outputs_us;
  uint64_t qnn_graph_execute_us;
  uint64_t qnn_shared_output_copyback_us;
  uint64_t qnn_graph_execute_count;
  uint64_t qnn_context_create_count;
  uint64_t qnn_context_create_from_binary_count;
  uint64_t qnn_graph_finalize_count;
  uint64_t qnn_custom_mem_dtype_float32_count;
  uint64_t qnn_custom_mem_dtype_float16_count;
  uint64_t qnn_custom_mem_dtype_other_count;
  uint64_t rpcmem_alloc_count;
  uint64_t rpcmem_free_count;
  uint64_t rpcmem_total_bytes;
  uint64_t custom_mem_addr_map_count;
  uint64_t custom_mem_addr_hit_count;
  uint64_t custom_mem_addr_miss_count;
  uint64_t qnn_pmd_hidden_slot_bypass_count;
  uint64_t qnn_shared_output_copyback_count;
  uint64_t qnn_shared_output_copyback_bytes;
} QnnExecuTorchAotDiagCounters;

enum QnnExecuTorchAotDiagCounter {
  kAotDiagQnnBackendExecute = 0,
  kAotDiagQnnRegisterMemCheck = 1,
  kAotDiagQnnRegisterMemFallback = 2,
  kAotDiagQnnRegisterMemCustomBaseHit = 3,
  kAotDiagQnnRegisterMemCustomBaseMiss = 4,
  kAotDiagQnnRegisterIonAttempt = 5,
  kAotDiagQnnRegisterCustomMemEntry = 6,
  kAotDiagQnnSharedBufferCurrentHit = 7,
  kAotDiagQnnSharedBufferPreregisteredHit = 8,
  kAotDiagQnnMemRegister = 9,
  kAotDiagQnnIonMemRegister = 10,
  kAotDiagQnnCustomMemRegister = 11,
  kAotDiagQnnSetMemHandle = 12,
  kAotDiagQnnFillDataBuffer = 13,
  kAotDiagQnnFillDataBufferBytes = 14,
  kAotDiagQnnGraphExecute = 15,
  kAotDiagQnnContextCreate = 16,
  kAotDiagQnnContextCreateFromBinary = 17,
  kAotDiagQnnGraphFinalize = 18,
  kAotDiagQnnCustomMemDtypeFloat32 = 19,
  kAotDiagQnnCustomMemDtypeFloat16 = 20,
  kAotDiagQnnCustomMemDtypeOther = 21,
  kAotDiagRpcmemAlloc = 22,
  kAotDiagRpcmemFree = 23,
  kAotDiagRpcmemTotalBytes = 24,
  kAotDiagCustomMemAddrMap = 25,
  kAotDiagCustomMemAddrHit = 26,
  kAotDiagCustomMemAddrMiss = 27,
  kAotDiagQnnPmdHiddenSlotBypass = 28,
  kAotDiagQnnSharedOutputCopyback = 29,
  kAotDiagQnnSharedOutputCopybackBytes = 30,
  kAotDiagQnnBackendExecuteUs = 31,
  kAotDiagQnnPrepareInputsUs = 32,
  kAotDiagQnnPrepareOutputsUs = 33,
  kAotDiagQnnGraphExecuteUs = 34,
  kAotDiagQnnSharedOutputCopybackUs = 35,
};

__attribute__((__visibility__("default"))) void QnnExecuTorchAotDiagSetEnabled(
    int enabled);
__attribute__((__visibility__("default"))) int QnnExecuTorchAotDiagIsEnabled();
__attribute__((__visibility__("default"))) void QnnExecuTorchAotDiagReset();
__attribute__((__visibility__("default"))) void QnnExecuTorchAotDiagAdd(
    int counter,
    uint64_t value);
__attribute__((__visibility__("default"))) void QnnExecuTorchAotDiagGet(
    QnnExecuTorchAotDiagCounters* counters);

#ifdef __cplusplus
}
#endif // __cplusplus
