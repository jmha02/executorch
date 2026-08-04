/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>

namespace {

struct AtomicAotDiagCounters {
  std::atomic<uint64_t> qnn_backend_execute_count{0};
  std::atomic<uint64_t> qnn_register_mem_check_count{0};
  std::atomic<uint64_t> qnn_register_mem_fallback_count{0};
  std::atomic<uint64_t> qnn_register_mem_custom_base_hit_count{0};
  std::atomic<uint64_t> qnn_register_mem_custom_base_miss_count{0};
  std::atomic<uint64_t> qnn_register_ion_attempt_count{0};
  std::atomic<uint64_t> qnn_register_custom_mem_entry_count{0};
  std::atomic<uint64_t> qnn_shared_buffer_current_hit_count{0};
  std::atomic<uint64_t> qnn_shared_buffer_preregistered_hit_count{0};
  std::atomic<uint64_t> qnn_mem_register_count{0};
  std::atomic<uint64_t> qnn_ion_mem_register_count{0};
  std::atomic<uint64_t> qnn_custom_mem_register_count{0};
  std::atomic<uint64_t> qnn_set_mem_handle_count{0};
  std::atomic<uint64_t> qnn_fill_data_buffer_count{0};
  std::atomic<uint64_t> qnn_fill_data_buffer_bytes{0};
  std::atomic<uint64_t> qnn_backend_execute_us{0};
  std::atomic<uint64_t> qnn_prepare_inputs_us{0};
  std::atomic<uint64_t> qnn_prepare_outputs_us{0};
  std::atomic<uint64_t> qnn_graph_execute_us{0};
  std::atomic<uint64_t> qnn_shared_output_copyback_us{0};
  std::atomic<uint64_t> qnn_graph_execute_count{0};
  std::atomic<uint64_t> qnn_context_create_count{0};
  std::atomic<uint64_t> qnn_context_create_from_binary_count{0};
  std::atomic<uint64_t> qnn_graph_finalize_count{0};
  std::atomic<uint64_t> qnn_custom_mem_dtype_float32_count{0};
  std::atomic<uint64_t> qnn_custom_mem_dtype_float16_count{0};
  std::atomic<uint64_t> qnn_custom_mem_dtype_other_count{0};
  std::atomic<uint64_t> rpcmem_alloc_count{0};
  std::atomic<uint64_t> rpcmem_free_count{0};
  std::atomic<uint64_t> rpcmem_total_bytes{0};
  std::atomic<uint64_t> custom_mem_addr_map_count{0};
  std::atomic<uint64_t> custom_mem_addr_hit_count{0};
  std::atomic<uint64_t> custom_mem_addr_miss_count{0};
  std::atomic<uint64_t> qnn_pmd_hidden_slot_bypass_count{0};
  std::atomic<uint64_t> qnn_shared_output_copyback_count{0};
  std::atomic<uint64_t> qnn_shared_output_copyback_bytes{0};
};

std::atomic<bool> g_aot_diag_enabled{false};
AtomicAotDiagCounters g_aot_diag_counters;

bool ShouldOmitBinarySectionWeightsUpdatesConfigForProbe() {
  const char* value = std::getenv(
      "EXECUTORCH_QNN_OMIT_BINARY_SECTION_WEIGHTS_UPDATES_CONFIG");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

uint64_t load_counter(const std::atomic<uint64_t>& counter) {
  return counter.load(std::memory_order_relaxed);
}

void reset_counter(std::atomic<uint64_t>& counter) {
  counter.store(0, std::memory_order_relaxed);
}

void add_counter(std::atomic<uint64_t>& counter, uint64_t value) {
  counter.fetch_add(value, std::memory_order_relaxed);
}

} // namespace

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

bool CompareExportedInput(
    const std::shared_ptr<TensorWrapper>& a,
    const std::shared_ptr<TensorWrapper>& b) {
  // Using the order of the nodes as external_id in AOT
  // to extract the right arg from *args at runtime
  int numA = std::stoi(a->GetName().substr(a->GetName().find('_') + 1));
  int numB = std::stoi(b->GetName().substr(b->GetName().find('_') + 1));
  return numA < numB;
}

int ExtractMutableBufferNumber(const std::string& name) {
  std::string prefix = "mutbuf_";
  size_t startPos = name.find(prefix);
  if (startPos != std::string::npos) {
    startPos += prefix.length();
    return std::stoi(name.substr(startPos));
  }
  return -1;
}

QnnManager::~QnnManager() {
  Destroy();
}

QnnManager::QnnManager(
    const QnnExecuTorchOptions* options,
    const QnnExecuTorchContextBinary& qnn_executorch_context_binary)
    : qnn_context_blob_(qnn_executorch_context_binary), options_(options) {
  QnnExecuTorchBackendType backend_type =
      options->backend_options()->backend_type();

  if (get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL) >=
      QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "soc_model in soc_info: %s",
        EnumNameQcomChipset(options_->soc_info()->soc_model()));
    QNN_EXECUTORCH_LOG_INFO(
        "backend_type: %s", EnumNameQnnExecuTorchBackendType(backend_type));
    QNN_EXECUTORCH_LOG_INFO(
        "library_path: %s", options->library_path()->str().c_str());
    QNN_EXECUTORCH_LOG_INFO("dump intermediate outputs: %s", IsTensorDump());
    QNN_EXECUTORCH_LOG_INFO(
        "log_level: %s",
        EnumNameQnnExecuTorchLogLevel(
            get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL)));
    QNN_EXECUTORCH_LOG_INFO(
        "profile_level: %s",
        EnumNameQnnExecuTorchProfileLevel(
            get_option(options_->profile_level(), QNN_RUNTIME_PROFILE_LEVEL)));
    QNN_EXECUTORCH_LOG_INFO(
        "the size of qnn context binary: %d",
        qnn_executorch_context_binary.nbytes);
    QNN_EXECUTORCH_LOG_INFO(
        "Is on-device graph construction: %d", options->online_prepare());
    QNN_EXECUTORCH_LOG_INFO(
        "Enable shared buffer: %d", options->shared_buffer());
    QNN_EXECUTORCH_LOG_INFO(
        "The number of op packages: %d",
        options_->op_package_options()->op_package_infos()->size());
  }

  backend_params_ptr_ = std::make_unique<BackendConfigParameters>();
  backend_bundle_ptr_ = std::make_shared<QnnBackendBundle>();

  qnn_dlc_manager_ =
      std::make_shared<QnnDlcManager>(qnn_context_blob_, options_);
}

Error QnnManager::RegisterMem(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  QnnExecuTorchAotDiagAdd(kAotDiagQnnRegisterMemCheck, 1);
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  // Not enable shared buffer
  if (!options_->shared_buffer()) {
    return Error::Internal;
  }

  if (backend_params_ptr_->qnn_mem_manager_ptr_ == nullptr) {
    QNN_EXECUTORCH_LOG_WARN(
        "Backend %s doesn't supported shared buffer.",
        EnumNameQnnExecuTorchBackendType(
            options_->backend_options()->backend_type()));
    return Error::Internal;
  }

  void* custom_mem_base = shared_buffer_manager.GetCustomMemBase(data_ptr);
  if (custom_mem_base != nullptr) {
    QnnExecuTorchAotDiagAdd(kAotDiagQnnRegisterMemCustomBaseHit, 1);
    return RegisterCustomMem(data_ptr, custom_mem_base, tensor_wrapper);
  }
  QnnExecuTorchAotDiagAdd(kAotDiagQnnRegisterMemCustomBaseMiss, 1);
  return RegisterIonMem(data_ptr, tensor_wrapper);
}

Error QnnManager::RegisterIonMem(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  QnnExecuTorchAotDiagAdd(kAotDiagQnnRegisterIonAttempt, 1);
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  if (!shared_buffer_manager.IsAllocated(data_ptr)) {
    // It means two scenarios here:
    // 1. the input and output partitioned graph
    // 2. Actually, user doesn't allocate shared buffer with
    // QnnExecuTorchAllocCustomMem API
    return Error::Internal;
  } else if (backend_params_ptr_->qnn_mem_manager_ptr_->IsRegistered(
                 tensor_wrapper->GetMemHandle(), data_ptr)) {
    if (get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL) >=
        QnnExecuTorchLogLevel::kLogLevelInfo)
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s has been registered shared memory.",
          tensor_wrapper->GetName().c_str());
    return Error::Ok;
  }

  int32_t mem_fd = shared_buffer_manager.MemToFd(data_ptr);
  if (mem_fd == -1) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor name %s is failed to get file descriptor.",
        tensor_wrapper->GetName().c_str());
    return Error::Internal;
  }
  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_mem_manager_ptr_->RegisterIonMem(
          tensor_wrapper, mem_fd, data_ptr) == Error::Ok,
      Internal,
      "Fail to register to shared memory.");

  return Error::Ok;
}

Error QnnManager::RegisterCustomMem(
    void* data_ptr,
    void* custom_mem_base,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  QnnExecuTorchAotDiagAdd(kAotDiagQnnRegisterCustomMemEntry, 1);
  if (backend_params_ptr_->qnn_mem_manager_ptr_->IsRegistered(
          tensor_wrapper->GetMemHandle(), data_ptr)) {
    QnnExecuTorchAotDiagAdd(kAotDiagQnnSharedBufferCurrentHit, 1);
    if (get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL) >=
        QnnExecuTorchLogLevel::kLogLevelInfo)
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s has been registered shared memory.",
          tensor_wrapper->GetName().c_str());
    return Error::Ok;
  }

  CustomMemTensorInfo info{
      custom_mem_base,
      data_ptr,
      static_cast<size_t>(
          static_cast<char*>(data_ptr) - static_cast<char*>(custom_mem_base)),
      tensor_wrapper->GetBytes(),
      tensor_wrapper->GetDims(),
      tensor_wrapper->GetRank(),
      qnn_dtype_to_scalar_type_[tensor_wrapper->GetDataType()]};

  Qnn_MemHandle_t pre_registered_handle =
      backend_params_ptr_->qnn_mem_manager_ptr_->GetPreRegisteredHandle(info);
  // If this memory block has already been registered, we can use it directly.
  // This applies when running llama in lookahead mode with the same AR-N model
  // handling both the prompt processor and the token generator.
  if (pre_registered_handle != nullptr) {
    QnnExecuTorchAotDiagAdd(kAotDiagQnnSharedBufferPreregisteredHit, 1);
    if (get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL) >=
        QnnExecuTorchLogLevel::kLogLevelInfo) {
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s found a pre-registered memHandle.",
          tensor_wrapper->GetName().c_str());
    }
    return backend_params_ptr_->qnn_mem_manager_ptr_->SetMemHandle(
        tensor_wrapper, data_ptr, pre_registered_handle);
  }

  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();

  size_t tensor_offset = info.pos;
  size_t total_custom_mem_size =
      shared_buffer_manager.GetAllocatedSize(custom_mem_base);

  int32_t mem_fd = shared_buffer_manager.MemToFd(custom_mem_base);
  // Note: If obtaining the file descriptor fails, it may be due to memory not
  // being released with QnnExecuTorchFreeCustomMem. In this situation, we could
  // consider adding a map to monitor it.
  if (mem_fd == -1) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor name %s failed to get file descriptor.",
        tensor_wrapper->GetName().c_str());
    return Error::Internal;
  }

  switch (tensor_wrapper->GetDataType()) {
    case QNN_DATATYPE_FLOAT_32:
      QnnExecuTorchAotDiagAdd(kAotDiagQnnCustomMemDtypeFloat32, 1);
      break;
    case QNN_DATATYPE_FLOAT_16:
      QnnExecuTorchAotDiagAdd(kAotDiagQnnCustomMemDtypeFloat16, 1);
      break;
    default:
      QnnExecuTorchAotDiagAdd(kAotDiagQnnCustomMemDtypeOther, 1);
      break;
  }

  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_mem_manager_ptr_->RegisterCustomMem(
          tensor_wrapper,
          mem_fd,
          data_ptr,
          total_custom_mem_size,
          tensor_offset,
          info) == Error::Ok,
      Internal,
      "Fail to register to shared memory.");

  return Error::Ok;
}

Error QnnManager::InitBackend() {
  // Get or create the shared backend bundle
  Error err = QnnBackendUnifiedRegistry::GetInstance().GetOrCreateBackendBundle(
      options_, backend_bundle_ptr_);
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok,
      Internal,
      "Fail to get or create shared Qnn backend bundle. Error code: %d",
      static_cast<int>(err));
  return Error::Ok;
}

Error QnnManager::InitContext(
    std::optional<std::vector<std::string>> graph_names) {
  if (backend_params_ptr_->backend_init_state_ ==
      BackendInitializeState::UNINITIALIZED) {
    QNN_EXECUTORCH_LOG_INFO(
        "Initialize Qnn backend "
        "parameters for Qnn executorch backend type %d",
        options_->backend_options()->backend_type());
    backend_params_ptr_ = QnnBackendFactory().Create(
        backend_bundle_ptr_->implementation.get(),
        backend_bundle_ptr_->qnn_backend_ptr.get(),
        backend_bundle_ptr_->qnn_device_ptr.get(),
        qnn_context_blob_,
        options_,
        qnn_dlc_manager_.get());
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_ != nullptr,
        Internal,
        "Failed to load Qnn backend.");
    // Note: For online_prepare or deserialization, the graph name will be
    // obtained from the binary.
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_backend_cache_ptr_->Configure(
            graph_names.value_or(std::vector<std::string>{})) == Error::Ok,
        Internal,
        "Fail to configure Qnn backend cache");
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_context_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn context");
    for (const std::string& graph_name :
         backend_params_ptr_->qnn_context_ptr_->GetGraphNames()) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_graph_ptr_->Configure(graph_name) ==
              Error::Ok,
          Internal,
          "Fail to configure Qnn graph");
    }

    backend_params_ptr_->backend_init_state_ =
        BackendInitializeState::INITIALIZED;
  }

  if (IsOnlinePrepare()) {
    // Check whether the QNN version supports the DLC format.
    Qnn_ApiVersion_t qnn_version = {QNN_VERSION_INIT};
    backend_bundle_ptr_->implementation->GetQnnInterface()
        .qnn_backend_get_api_version(&qnn_version);

    ET_CHECK_OR_RETURN_ERROR(
        qnn_dlc_manager_->SetUpDlcEnvironment(
            qnn_version.coreApiVersion,
            graph_names.value_or(std::vector<std::string>{})) == Error::Ok,
        Internal,
        "Fail to setup Dlc environment");
  }
  return Error::Ok;
}

Error QnnManager::InitContextCache() {
  if (backend_params_ptr_->backend_init_state_ ==
      BackendInitializeState::UNINITIALIZED) {
    QNN_EXECUTORCH_LOG_INFO(
        "Initialize Qnn backend "
        "parameters for Qnn executorch backend type %d",
        options_->backend_options()->backend_type());
    backend_params_ptr_ = QnnBackendFactory().Create(
        backend_bundle_ptr_->implementation.get(),
        backend_bundle_ptr_->qnn_backend_ptr.get(),
        backend_bundle_ptr_->qnn_device_ptr.get(),
        qnn_context_blob_,
        options_,
        qnn_dlc_manager_.get());
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_ != nullptr,
        Internal,
        "Failed to load Qnn backend.");
    // Note: For online_prepare or deserialization, the graph name will be
    // obtained from the binary.
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_backend_cache_ptr_->Configure({}) == Error::Ok,
        Internal,
        "Fail to configure Qnn backend cache");

    backend_params_ptr_->backend_init_state_ =
        BackendInitializeState::INITIALIZED;
  }
  return Error::Ok;
}

Error QnnManager::AllocateTensor(const std::string& graph_name) {
  std::vector<Qnn_Tensor_t> input_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphInputs(graph_name);
  std::vector<Qnn_Tensor_t> output_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphOutputs(graph_name);

  // Mapping memory address for the input and output of mutable buffer
  std::unordered_map<int, const void*> mutable_buffer_id_to_memory_map;

  for (auto& tensor : input_tensors) {
    std::shared_ptr<TensorWrapper> tensor_wrapper = CreateTensorWrapper(tensor);
    tensor_wrapper->UpdateQnnTensorMeta(tensor);

    int mutable_buffer_id =
        ExtractMutableBufferNumber(tensor_wrapper->GetName());
    if (mutable_buffer_id != -1) {
      // Delegate maintains the memory for mutable buffer
      tensor_wrapper->AllocateDataBuffer();
      mutable_buffer_id_to_memory_map[mutable_buffer_id] =
          tensor_wrapper->GetStaticTensorData();
    }
    input_tensors_[graph_name].emplace_back(std::move(tensor_wrapper));
  }
  // Always sort by input_{external_id}_ so GetGraphInputs matches call_delegate
  // args order. Skipping this for context binaries left large HTP-fused graphs
  // with mixed-rank inputs (e.g. [2,8] tokens + [2,1,8,8] masks) misaligned →
  // SetDims "2 vs 4" and DSP Error 6004 batch (2, 8).
  std::sort(
      input_tensors_[graph_name].begin(),
      input_tensors_[graph_name].end(),
      CompareExportedInput);
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    std::shared_ptr<TensorWrapper> tensor_wrapper =
        CreateTensorWrapper(output_tensors[i]);
    tensor_wrapper->UpdateQnnTensorMeta(output_tensors[i]);
    const std::string& tensor_name = tensor_wrapper->GetName();
    // this is required by identifying shared buffer mechanism
    // info might be missed if context binary came from qnn_converter
    if (options_->is_from_context_binary() &&
        tensor_name.find("output_") == std::string::npos) {
      tensor_wrapper->SetName("output_" + tensor_name);
    }
    if (IsTensorDump()) {
      tensor_wrapper->AllocateDataBuffer();
    }
    int mutable_buffer_id =
        ExtractMutableBufferNumber(tensor_wrapper->GetName());
    if (mutable_buffer_id != -1 &&
        mutable_buffer_id_to_memory_map.find(mutable_buffer_id) !=
            mutable_buffer_id_to_memory_map.end()) {
      // Fill the same memory for I/O of mutable buffer
      tensor_wrapper->FillDataBuffer(
          mutable_buffer_id_to_memory_map[mutable_buffer_id]);
    }
    output_tensors_[graph_name].emplace_back(std::move(tensor_wrapper));
  }
  return Error::Ok;
}

Error QnnManager::AllocateTensor(
    const std::string& graph_name,
    std::vector<std::shared_ptr<TensorWrapper>>& inputs,
    std::vector<std::shared_ptr<TensorWrapper>>& outputs) {
  input_tensors_[graph_name] = std::move(inputs);
  // TODO: suuport per-tensor dump in online prepare mode
  //       should be achievable with some pre-process
  if (!options_->is_from_context_binary()) {
    std::sort(
        input_tensors_[graph_name].begin(),
        input_tensors_[graph_name].end(),
        CompareExportedInput);
  }
  output_tensors_[graph_name] = std::move(outputs);
  return Error::Ok;
}

Error QnnManager::Execute(
    const std::string& graph_name,
    const std::vector<Qnn_Tensor_t>& input_tensor_structs,
    std::vector<Qnn_Tensor_t>& output_tensor_structs,
    executorch::runtime::EventTracer* event_tracer) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  QnnExecuTorchAotDiagAdd(kAotDiagQnnGraphExecute, 1);
  error = backend_params_ptr_->qnn_graph_ptr_->GraphExecute(
      graph_name, input_tensor_structs, output_tensor_structs);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "qnn_graph_execute failed. Error %d", QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  if (IsTensorDump()) {
    // TODO: Need to handle the graph which is partitioned.
    // Maybe we could use graph name.
    for (std::size_t out_idx = 0; out_idx < output_tensor_structs.size();
         ++out_idx) {
      const Qnn_Tensor_t& output_tensor = output_tensor_structs[out_idx];
      std::vector<executorch::aten::SizesType> sizes(
          QNN_TENSOR_VER_PTR(output_tensor)->dimensions,
          QNN_TENSOR_VER_PTR(output_tensor)->dimensions +
              QNN_TENSOR_VER_PTR(output_tensor)->rank);

      // Compute contiguous strides from sizes (e.g. [2,3,4] -> [12,4,1]).
      std::vector<executorch::aten::StridesType> stride_size(sizes.size());
      if (!sizes.empty()) {
        stride_size.back() = 1;
        for (int i = sizes.size() - 2; i >= 0; --i) {
          stride_size[i] = stride_size[i + 1] * sizes[i + 1];
        }
      }
      // Avoid using from_blob as it significantly increases shared library
      // size.
      executorch::aten::TensorImpl tensor_impl(
          qnn_dtype_to_scalar_type_[QNN_TENSOR_VER_PTR(output_tensor)
                                        ->dataType],
          sizes.size(),
          sizes.data(),
          QNN_TENSOR_VER_PTR(output_tensor)->clientBuf.data,
          nullptr,
          stride_size.data());

      executorch::runtime::event_tracer_log_output_delegate<
          executorch::aten::Tensor>(
          event_tracer,
          QNN_TENSOR_VER_PTR(output_tensor)->name,
          /*delegate_debug_id=*/
          static_cast<executorch::runtime::DebugHandle>(-1),
          executorch::aten::Tensor(&tensor_impl));
    }
  }

  return Error::Ok;
}

Error QnnManager::ProfileExecuteData(
    const std::string& graph_name,
    executorch::runtime::EventTracer* event_tracer) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (get_option(options_->profile_level(), QNN_RUNTIME_PROFILE_LEVEL) !=
      QnnExecuTorchProfileLevel::kProfileOff) {
    error = backend_params_ptr_->qnn_graph_ptr_->ProfileExecuteData(
        graph_name, event_tracer);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          " Failed to profile. Error %d", QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  return Error::Ok;
}

void QnnManager::Destroy() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  backend_bundle_ptr_.reset(new QnnBackendBundle());
  qnn_dlc_manager_->Destroy();
}

void QnnManager::DestroyContext() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  qnn_dlc_manager_->Destroy();
}

bool QnnManager::IsNodeSupportedByBackend(
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& param : op_wrapper->GetParams()) {
      // unused?
      // auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (param->PopulateQnnParam() != Error::Ok) {
        QNN_EXECUTORCH_LOG_WARN(
            "Qnn Backend op validation failed "
            "with PopulateQnnParam: %d",
            QNN_GET_ERROR_CODE(error));
        return false;
      }
    }

    error = backend_bundle_ptr_->qnn_backend_ptr->BackendValidateOpConfig(
        op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Qnn Backend op validation failed with error: %d",
          QNN_GET_ERROR_CODE(error));

      return false;
    }
  }
  return true;
}

Error QnnManager::GetContextBinary(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  if (IsOnlinePrepare() &&
      qnn_dlc_manager_->backend_params_ptr_->qnn_context_ptr_.get() !=
          nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_dlc_manager_->backend_params_ptr_->qnn_context_ptr_
                ->GetContextBinary(qnn_executorch_context_binary) == Error::Ok,
        Internal,
        "Fail to get context binary.");
  }

  else {
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_context_ptr_->GetContextBinary(
            qnn_executorch_context_binary) == Error::Ok,
        Internal,
        "Fail to get context binary.");
  }
  return Error::Ok;
}

Error QnnManager::GetUpdatableWeightsBinarySection(
    const std::string& graph_name,
    std::vector<uint8_t>& section) {
  section.clear();
  updatable_weights_lifecycle_trace_.section_size_status = QNN_ERROR_UNDEFINED;
  updatable_weights_lifecycle_trace_.section_extraction_status =
      QNN_ERROR_UNDEFINED;
  updatable_weights_lifecycle_trace_.section_bytes = 0;
  ET_CHECK_OR_RETURN_ERROR(
      backend_bundle_ptr_ != nullptr &&
          backend_bundle_ptr_->implementation != nullptr &&
          backend_params_ptr_ != nullptr &&
          backend_params_ptr_->qnn_context_ptr_ != nullptr &&
          backend_params_ptr_->qnn_graph_ptr_ != nullptr,
      Internal,
      "QNN source context is not initialized for binary-section extraction");

  const QnnInterface& qnn_interface =
      backend_bundle_ptr_->implementation->GetQnnInterface();
  ET_CHECK_OR_RETURN_ERROR(
      qnn_interface.HasContextGetBinarySectionSize() &&
          qnn_interface.HasContextGetBinarySection(),
      Internal,
      "QNN backend does not expose binary-section extraction functions");

  Qnn_GraphHandle_t graph_handle =
      backend_params_ptr_->qnn_graph_ptr_->GetHandle(graph_name);
  ET_CHECK_OR_RETURN_ERROR(
      graph_handle != nullptr,
      Internal,
      "QNN graph %s is unavailable for binary-section extraction",
      graph_name.c_str());

  Qnn_ContextBinarySize_t section_size = 0;
  Qnn_ErrorHandle_t error = qnn_interface.qnn_context_get_binary_section_size(
      backend_params_ptr_->qnn_context_ptr_->GetHandle(),
      graph_handle,
      QNN_CONTEXT_SECTION_UPDATABLE_WEIGHTS,
      &section_size);
  updatable_weights_lifecycle_trace_.section_size_status =
      QNN_GET_ERROR_CODE(error);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QNN updatable-weights section size failed. Error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  ET_CHECK_OR_RETURN_ERROR(
      section_size > 0,
      Internal,
      "QNN updatable-weights section has zero bytes");

  section.resize(section_size);
  QnnContext_Buffer_t buffer{};
  buffer.version = QNN_CONTEXT_BUFFER_VERSION_1;
  buffer.v1.memType = QNN_CONTEXTMEMTYPE_RAW;
  buffer.v1.binaryBuf.data = section.data();
  buffer.v1.binaryBuf.dataSize = section.size();
  Qnn_ContextBinarySize_t written_size = 0;
  error = qnn_interface.qnn_context_get_binary_section(
      backend_params_ptr_->qnn_context_ptr_->GetHandle(),
      graph_handle,
      QNN_CONTEXT_SECTION_UPDATABLE_WEIGHTS,
      &buffer,
      &written_size,
      nullptr,
      nullptr);
  updatable_weights_lifecycle_trace_.section_extraction_status =
      QNN_GET_ERROR_CODE(error);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QNN updatable-weights section extraction failed. Error: %d",
        QNN_GET_ERROR_CODE(error));
    section.clear();
    return Error::Internal;
  }
  ET_CHECK_OR_RETURN_ERROR(
      written_size > 0 && written_size <= section.size(),
      Internal,
      "QNN updatable-weights section wrote invalid byte count %llu of %llu",
      static_cast<unsigned long long>(written_size),
      static_cast<unsigned long long>(section.size()));
  section.resize(written_size);
  updatable_weights_lifecycle_trace_.section_bytes = section.size();
  return Error::Ok;
}

Error QnnManager::UpdateFirstUpdatableStaticTensorAndRefinalize(
    const std::string& graph_name) {
  ET_CHECK_OR_RETURN_ERROR(
      backend_bundle_ptr_ != nullptr &&
          backend_bundle_ptr_->implementation != nullptr &&
          backend_params_ptr_ != nullptr &&
          backend_params_ptr_->qnn_graph_ptr_ != nullptr,
      Internal,
      "QNN source graph is not initialized for update/refinalize");

  const QnnInterface& qnn_interface =
      backend_bundle_ptr_->implementation->GetQnnInterface();
  updatable_weights_lifecycle_trace_.tensor_update_graph_tensors_available =
      qnn_interface.HasTensorUpdateGraphTensors();
  updatable_weights_lifecycle_trace_.graph_finalize_available = true;
  ET_CHECK_OR_RETURN_ERROR(
      qnn_interface.HasTensorUpdateGraphTensors(),
      Internal,
      "QNN backend does not expose tensorUpdateGraphTensors");

  const auto tensor_it = updateable_static_tensors_.find(graph_name);
  ET_CHECK_OR_RETURN_ERROR(
      tensor_it != updateable_static_tensors_.end() && !tensor_it->second.empty(),
      Internal,
      "QNN graph %s has no tracked UPDATEABLE_STATIC tensor",
      graph_name.c_str());
  const std::shared_ptr<TensorWrapper>& tensor_wrapper = tensor_it->second.front();
  ET_CHECK_OR_RETURN_ERROR(
      tensor_wrapper->HasInitialPayload() && tensor_wrapper->GetBytes() > 0,
      Internal,
      "UPDATEABLE_STATIC tensor %s has no initialized payload",
      tensor_wrapper->GetName().c_str());

  Qnn_Tensor_t updated_tensor = tensor_wrapper->CloneTensorStruct();
  std::vector<uint8_t> updated_payload(tensor_wrapper->GetBytes());
  std::memcpy(
      updated_payload.data(),
      tensor_wrapper->GetStaticTensorData(),
      updated_payload.size());
  updated_payload.front() ^= 0x01;
  QNN_TENSOR_VER_PTR(updated_tensor)->clientBuf.data = updated_payload.data();
  QNN_TENSOR_VER_PTR(updated_tensor)->clientBuf.dataSize = updated_payload.size();

  updatable_weights_lifecycle_trace_.tensor_name = tensor_wrapper->GetName();
  updatable_weights_lifecycle_trace_.tensor_id =
      QNN_TENSOR_VER_PTR(updated_tensor)->id;
  updatable_weights_lifecycle_trace_.tensor_dims.assign(
      tensor_wrapper->GetDims(),
      tensor_wrapper->GetDims() + tensor_wrapper->GetRank());
  updatable_weights_lifecycle_trace_.payload_bytes = updated_payload.size();
  uint64_t payload_hash = 1469598103934665603ULL;
  for (const uint8_t byte : updated_payload) {
    payload_hash ^= byte;
    payload_hash *= 1099511628211ULL;
  }
  updatable_weights_lifecycle_trace_.payload_fnv1a64 = payload_hash;

  const Qnn_Tensor_t* tensors[] = {&updated_tensor};
  Qnn_GraphHandle_t graph_handle =
      backend_params_ptr_->qnn_graph_ptr_->GetHandle(graph_name);
  ET_CHECK_OR_RETURN_ERROR(
      graph_handle != nullptr,
      Internal,
      "QNN graph %s is unavailable for update/refinalize",
      graph_name.c_str());
  Qnn_ErrorHandle_t error = qnn_interface.qnn_tensor_update_graph_tensors(
      graph_handle, tensors, 1);
  updatable_weights_lifecycle_trace_.tensor_update_status =
      QNN_GET_ERROR_CODE(error);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QNN updateable tensor update failed. Error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  QnnExecuTorchAotDiagAdd(kAotDiagQnnGraphFinalize, 1);
  error = backend_params_ptr_->qnn_graph_ptr_->GraphFinalize(graph_name);
  updatable_weights_lifecycle_trace_.graph_refinalize_status =
      QNN_GET_ERROR_CODE(error);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QNN updateable graph re-finalize failed. Error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}

Error QnnManager::UpdateAllUpdatableStaticTensorsAndRefinalize(
    const std::string& graph_name) {
  ET_CHECK_OR_RETURN_ERROR(
      backend_bundle_ptr_ != nullptr &&
          backend_bundle_ptr_->implementation != nullptr &&
          backend_params_ptr_ != nullptr &&
          backend_params_ptr_->qnn_graph_ptr_ != nullptr,
      Internal,
      "QNN source graph is not initialized for batched update/refinalize");

  const QnnInterface& qnn_interface =
      backend_bundle_ptr_->implementation->GetQnnInterface();
  updatable_weights_lifecycle_trace_.tensor_update_graph_tensors_available =
      qnn_interface.HasTensorUpdateGraphTensors();
  updatable_weights_lifecycle_trace_.graph_finalize_available = true;
  ET_CHECK_OR_RETURN_ERROR(
      qnn_interface.HasTensorUpdateGraphTensors(),
      Internal,
      "QNN backend does not expose tensorUpdateGraphTensors");

  const auto tensor_it = updateable_static_tensors_.find(graph_name);
  ET_CHECK_OR_RETURN_ERROR(
      tensor_it != updateable_static_tensors_.end() &&
          !tensor_it->second.empty(),
      Internal,
      "QNN graph %s has no tracked UPDATEABLE_STATIC tensor",
      graph_name.c_str());

  std::vector<Qnn_Tensor_t> updated_tensors;
  std::vector<std::vector<uint8_t>> updated_payloads;
  std::vector<const Qnn_Tensor_t*> tensor_ptrs;
  updated_tensors.reserve(tensor_it->second.size());
  updated_payloads.reserve(tensor_it->second.size());
  tensor_ptrs.reserve(tensor_it->second.size());
  updatable_weights_lifecycle_trace_.updated_tensors.clear();

  for (size_t tensor_index = 0; tensor_index < tensor_it->second.size();
       ++tensor_index) {
    const std::shared_ptr<TensorWrapper>& tensor_wrapper =
        tensor_it->second[tensor_index];
    ET_CHECK_OR_RETURN_ERROR(
        tensor_wrapper->HasInitialPayload() && tensor_wrapper->GetBytes() > 0,
        Internal,
        "UPDATEABLE_STATIC tensor %s has no initialized payload",
        tensor_wrapper->GetName().c_str());

    updated_tensors.push_back(tensor_wrapper->CloneTensorStruct());
    updated_payloads.emplace_back(tensor_wrapper->GetBytes());
    std::vector<uint8_t>& updated_payload = updated_payloads.back();
    std::memcpy(
        updated_payload.data(),
        tensor_wrapper->GetStaticTensorData(),
        updated_payload.size());
    updated_payload[tensor_index % updated_payload.size()] ^=
        static_cast<uint8_t>(tensor_index + 1);
    QNN_TENSOR_VER_PTR(updated_tensors.back())->clientBuf.data =
        updated_payload.data();
    QNN_TENSOR_VER_PTR(updated_tensors.back())->clientBuf.dataSize =
        updated_payload.size();
    tensor_ptrs.push_back(&updated_tensors.back());

    UpdatableWeightsLifecycleTrace::TensorUpdateTrace trace;
    trace.tensor_name = tensor_wrapper->GetName();
    trace.tensor_id = QNN_TENSOR_VER_PTR(updated_tensors.back())->id;
    trace.tensor_dims.assign(
        tensor_wrapper->GetDims(),
        tensor_wrapper->GetDims() + tensor_wrapper->GetRank());
    trace.payload_bytes = updated_payload.size();
    trace.payload_fnv1a64 = 1469598103934665603ULL;
    for (const uint8_t byte : updated_payload) {
      trace.payload_fnv1a64 ^= byte;
      trace.payload_fnv1a64 *= 1099511628211ULL;
    }
    updatable_weights_lifecycle_trace_.updated_tensors.push_back(
        std::move(trace));
  }

  const auto& first_trace = updatable_weights_lifecycle_trace_.updated_tensors.front();
  updatable_weights_lifecycle_trace_.tensor_name = first_trace.tensor_name;
  updatable_weights_lifecycle_trace_.tensor_id = first_trace.tensor_id;
  updatable_weights_lifecycle_trace_.tensor_dims = first_trace.tensor_dims;
  updatable_weights_lifecycle_trace_.payload_bytes = first_trace.payload_bytes;
  updatable_weights_lifecycle_trace_.payload_fnv1a64 =
      first_trace.payload_fnv1a64;

  Qnn_GraphHandle_t graph_handle =
      backend_params_ptr_->qnn_graph_ptr_->GetHandle(graph_name);
  ET_CHECK_OR_RETURN_ERROR(
      graph_handle != nullptr,
      Internal,
      "QNN graph %s is unavailable for batched update/refinalize",
      graph_name.c_str());
  Qnn_ErrorHandle_t error = qnn_interface.qnn_tensor_update_graph_tensors(
      graph_handle, tensor_ptrs.data(), tensor_ptrs.size());
  updatable_weights_lifecycle_trace_.tensor_update_status =
      QNN_GET_ERROR_CODE(error);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QNN batched updateable tensor update failed. Error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  QnnExecuTorchAotDiagAdd(kAotDiagQnnGraphFinalize, 1);
  error = backend_params_ptr_->qnn_graph_ptr_->GraphFinalize(graph_name);
  updatable_weights_lifecycle_trace_.graph_refinalize_status =
      QNN_GET_ERROR_CODE(error);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QNN batched updateable graph re-finalize failed. Error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}

Error QnnManager::CompileDlc() {
  Qnn_ErrorHandle_t error;
  auto qnn_dlc_graph_info = qnn_dlc_manager_->GetQnnDlcGraphInfoPtr();
  uint32_t qnn_dlc_graph_info_num = qnn_dlc_manager_->GetQnnDlcGraphInfoNum();
  for (uint32_t i = 0; i < qnn_dlc_graph_info_num; ++i) {
    auto& graphInfo = (*qnn_dlc_graph_info)[i];
    backend_params_ptr_->qnn_graph_ptr_->SetGraphHandle(
        graphInfo.graphName, graphInfo.graph);
    QnnExecuTorchAotDiagAdd(kAotDiagQnnGraphFinalize, 1);
    error =
        backend_params_ptr_->qnn_graph_ptr_->GraphFinalize(graphInfo.graphName);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to finalize Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }

    std::vector<std::shared_ptr<TensorWrapper>> graph_inputs, graph_outputs,
        tensors;

    // Mapping memory address for the input and output of mutable buffer
    std::unordered_map<int, const void*> mutable_buffer_id_to_memory_map;
    for (uint32_t i = 0; i < graphInfo.numInputTensors; ++i) {
      auto tw = CreateTensorWrapper(graphInfo.inputTensors[i]);
      tw->UpdateQnnTensorMeta(graphInfo.inputTensors[i]);

      int mutable_buffer_id = ExtractMutableBufferNumber(tw->GetName());
      if (mutable_buffer_id != -1) {
        // Delegate maintains the memory for mutable buffer
        tw->AllocateDataBuffer();
        mutable_buffer_id_to_memory_map[mutable_buffer_id] =
            tw->GetStaticTensorData();
      }
      graph_inputs.push_back(tw);
    }
    for (uint32_t i = 0; i < graphInfo.numOutputTensors; ++i) {
      auto tw = CreateTensorWrapper(graphInfo.outputTensors[i]);
      tw->UpdateQnnTensorMeta(graphInfo.outputTensors[i]);
      int mutable_buffer_id = ExtractMutableBufferNumber(tw->GetName());
      if (mutable_buffer_id != -1 &&
          mutable_buffer_id_to_memory_map.find(mutable_buffer_id) !=
              mutable_buffer_id_to_memory_map.end()) {
        // Fill the same memory for I/O of mutable buffer
        tw->FillDataBuffer(mutable_buffer_id_to_memory_map[mutable_buffer_id]);
      }
      graph_outputs.push_back(tw);
    }

    ET_CHECK_OR_RETURN_ERROR(
        AllocateTensor(graphInfo.graphName, graph_inputs, graph_outputs) ==
            Error::Ok,
        Internal,
        "Fail to allocate tensor for Dlc with graph_name: %s",
        graphInfo.graphName);
  }

  return Error::Ok;
}

Error QnnManager::Compile(
    const std::string& graph_name,
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  QnnGraph* qnn_graph_ptr = backend_params_ptr_->qnn_graph_ptr_.get();

  if (IsOnlinePrepare() &&
      qnn_dlc_manager_->backend_params_ptr_->qnn_graph_ptr_.get() != nullptr) {
    qnn_graph_ptr = qnn_dlc_manager_->backend_params_ptr_->qnn_graph_ptr_.get();
  }
  bool has_updateable_static_tensor = false;
  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& tensor_wrapper : op_wrapper->GetInputTensors()) {
      has_updateable_static_tensor = has_updateable_static_tensor ||
          tensor_wrapper->GetTensorType() == QNN_TENSOR_TYPE_UPDATEABLE_STATIC;
      ET_CHECK_OR_RETURN_ERROR(
          qnn_graph_ptr->EnsureTensorInQnnGraph(graph_name, tensor_wrapper) ==
              Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
      if (tensor_wrapper->GetTensorType() ==
          QNN_TENSOR_TYPE_UPDATEABLE_STATIC) {
        auto& tracked_tensors = updateable_static_tensors_[graph_name];
        const auto duplicate = std::find_if(
            tracked_tensors.begin(),
            tracked_tensors.end(),
            [&tensor_wrapper](const std::shared_ptr<TensorWrapper>& tracked) {
              return tracked->GetName() == tensor_wrapper->GetName();
            });
        if (duplicate == tracked_tensors.end()) {
          tracked_tensors.push_back(tensor_wrapper);
        }
      }
    }
    for (const auto& tensor_wrapper : op_wrapper->GetOutputTensors()) {
      has_updateable_static_tensor = has_updateable_static_tensor ||
          tensor_wrapper->GetTensorType() == QNN_TENSOR_TYPE_UPDATEABLE_STATIC;
      ET_CHECK_OR_RETURN_ERROR(
          qnn_graph_ptr->EnsureTensorInQnnGraph(graph_name, tensor_wrapper) ==
              Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }
    for (const auto& param : op_wrapper->GetParams()) {
      auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (p_tensor_param != nullptr) {
        has_updateable_static_tensor = has_updateable_static_tensor ||
            p_tensor_param->GetTensorWrapper()->GetTensorType() ==
            QNN_TENSOR_TYPE_UPDATEABLE_STATIC;
        ET_CHECK_OR_RETURN_ERROR(
            qnn_graph_ptr->EnsureTensorInQnnGraph(
                graph_name, p_tensor_param->GetTensorWrapper()) == Error::Ok,
            Internal,
            "Param tensor name %s isn't added to Qnn Graph",
            p_tensor_param->GetName().c_str());
      }
      ET_CHECK_OR_RETURN_ERROR(
          param->PopulateQnnParam() == Error::Ok,
          Internal,
          "Fail to configure Qnn backend");
    }

    error = qnn_graph_ptr->GraphAddNode(graph_name, op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to add node to Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  const bool binary_section_weights_updates_config_enabled =
      has_updateable_static_tensor &&
      !ShouldOmitBinarySectionWeightsUpdatesConfigForProbe();
  updatable_weights_lifecycle_trace_
      .binary_section_weights_updates_config_enabled =
      binary_section_weights_updates_config_enabled;
  if (binary_section_weights_updates_config_enabled) {
    error = qnn_graph_ptr->EnableBinarySectionWeightUpdates(graph_name);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to enable QNN binary-section weight updates. Error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  QnnExecuTorchAotDiagAdd(kAotDiagQnnGraphFinalize, 1);
  error = qnn_graph_ptr->GraphFinalize(graph_name);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to finalize Qnn Graph with error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
void* QnnExecuTorchAllocCustomMem(size_t bytes, size_t alignment) {
  void* buffer_ptr =
      executorch::backends::qnn::SharedBuffer::GetSharedBufferManager()
          .AllocMem(bytes, alignment);
  return buffer_ptr;
}

void QnnExecuTorchFreeCustomMem(void* buffer_ptr) {
  executorch::backends::qnn::SharedBuffer::GetSharedBufferManager().FreeMem(
      buffer_ptr);
}

void QnnExecuTorchAddCustomMemTensorAddr(void* tensor_addr, void* custom_mem) {
  executorch::backends::qnn::SharedBuffer::GetSharedBufferManager()
      .AddCusomMemTensorAddr(tensor_addr, custom_mem);
}

void QnnExecuTorchAotDiagSetEnabled(int enabled) {
  g_aot_diag_enabled.store(enabled != 0, std::memory_order_relaxed);
}

int QnnExecuTorchAotDiagIsEnabled() {
  return g_aot_diag_enabled.load(std::memory_order_relaxed) ? 1 : 0;
}

void QnnExecuTorchAotDiagReset() {
  reset_counter(g_aot_diag_counters.qnn_backend_execute_count);
  reset_counter(g_aot_diag_counters.qnn_register_mem_check_count);
  reset_counter(g_aot_diag_counters.qnn_register_mem_fallback_count);
  reset_counter(g_aot_diag_counters.qnn_register_mem_custom_base_hit_count);
  reset_counter(g_aot_diag_counters.qnn_register_mem_custom_base_miss_count);
  reset_counter(g_aot_diag_counters.qnn_register_ion_attempt_count);
  reset_counter(g_aot_diag_counters.qnn_register_custom_mem_entry_count);
  reset_counter(g_aot_diag_counters.qnn_shared_buffer_current_hit_count);
  reset_counter(g_aot_diag_counters.qnn_shared_buffer_preregistered_hit_count);
  reset_counter(g_aot_diag_counters.qnn_mem_register_count);
  reset_counter(g_aot_diag_counters.qnn_ion_mem_register_count);
  reset_counter(g_aot_diag_counters.qnn_custom_mem_register_count);
  reset_counter(g_aot_diag_counters.qnn_set_mem_handle_count);
  reset_counter(g_aot_diag_counters.qnn_fill_data_buffer_count);
  reset_counter(g_aot_diag_counters.qnn_fill_data_buffer_bytes);
  reset_counter(g_aot_diag_counters.qnn_backend_execute_us);
  reset_counter(g_aot_diag_counters.qnn_prepare_inputs_us);
  reset_counter(g_aot_diag_counters.qnn_prepare_outputs_us);
  reset_counter(g_aot_diag_counters.qnn_graph_execute_us);
  reset_counter(g_aot_diag_counters.qnn_shared_output_copyback_us);
  reset_counter(g_aot_diag_counters.qnn_graph_execute_count);
  reset_counter(g_aot_diag_counters.qnn_context_create_count);
  reset_counter(g_aot_diag_counters.qnn_context_create_from_binary_count);
  reset_counter(g_aot_diag_counters.qnn_graph_finalize_count);
  reset_counter(g_aot_diag_counters.qnn_custom_mem_dtype_float32_count);
  reset_counter(g_aot_diag_counters.qnn_custom_mem_dtype_float16_count);
  reset_counter(g_aot_diag_counters.qnn_custom_mem_dtype_other_count);
  reset_counter(g_aot_diag_counters.rpcmem_alloc_count);
  reset_counter(g_aot_diag_counters.rpcmem_free_count);
  reset_counter(g_aot_diag_counters.rpcmem_total_bytes);
  reset_counter(g_aot_diag_counters.custom_mem_addr_map_count);
  reset_counter(g_aot_diag_counters.custom_mem_addr_hit_count);
  reset_counter(g_aot_diag_counters.custom_mem_addr_miss_count);
  reset_counter(g_aot_diag_counters.qnn_pmd_hidden_slot_bypass_count);
  reset_counter(g_aot_diag_counters.qnn_shared_output_copyback_count);
  reset_counter(g_aot_diag_counters.qnn_shared_output_copyback_bytes);
}

void QnnExecuTorchAotDiagAdd(int counter, uint64_t value) {
  if (!g_aot_diag_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  switch (counter) {
    case kAotDiagQnnBackendExecute:
      add_counter(g_aot_diag_counters.qnn_backend_execute_count, value);
      break;
    case kAotDiagQnnRegisterMemCheck:
      add_counter(g_aot_diag_counters.qnn_register_mem_check_count, value);
      break;
    case kAotDiagQnnRegisterMemFallback:
      add_counter(g_aot_diag_counters.qnn_register_mem_fallback_count, value);
      break;
    case kAotDiagQnnRegisterMemCustomBaseHit:
      add_counter(
          g_aot_diag_counters.qnn_register_mem_custom_base_hit_count, value);
      break;
    case kAotDiagQnnRegisterMemCustomBaseMiss:
      add_counter(
          g_aot_diag_counters.qnn_register_mem_custom_base_miss_count, value);
      break;
    case kAotDiagQnnRegisterIonAttempt:
      add_counter(g_aot_diag_counters.qnn_register_ion_attempt_count, value);
      break;
    case kAotDiagQnnRegisterCustomMemEntry:
      add_counter(g_aot_diag_counters.qnn_register_custom_mem_entry_count, value);
      break;
    case kAotDiagQnnSharedBufferCurrentHit:
      add_counter(g_aot_diag_counters.qnn_shared_buffer_current_hit_count, value);
      break;
    case kAotDiagQnnSharedBufferPreregisteredHit:
      add_counter(
          g_aot_diag_counters.qnn_shared_buffer_preregistered_hit_count, value);
      break;
    case kAotDiagQnnMemRegister:
      add_counter(g_aot_diag_counters.qnn_mem_register_count, value);
      break;
    case kAotDiagQnnIonMemRegister:
      add_counter(g_aot_diag_counters.qnn_ion_mem_register_count, value);
      break;
    case kAotDiagQnnCustomMemRegister:
      add_counter(g_aot_diag_counters.qnn_custom_mem_register_count, value);
      break;
    case kAotDiagQnnSetMemHandle:
      add_counter(g_aot_diag_counters.qnn_set_mem_handle_count, value);
      break;
    case kAotDiagQnnFillDataBuffer:
      add_counter(g_aot_diag_counters.qnn_fill_data_buffer_count, value);
      break;
    case kAotDiagQnnFillDataBufferBytes:
      add_counter(g_aot_diag_counters.qnn_fill_data_buffer_bytes, value);
      break;
    case kAotDiagQnnBackendExecuteUs:
      add_counter(g_aot_diag_counters.qnn_backend_execute_us, value);
      break;
    case kAotDiagQnnPrepareInputsUs:
      add_counter(g_aot_diag_counters.qnn_prepare_inputs_us, value);
      break;
    case kAotDiagQnnPrepareOutputsUs:
      add_counter(g_aot_diag_counters.qnn_prepare_outputs_us, value);
      break;
    case kAotDiagQnnGraphExecuteUs:
      add_counter(g_aot_diag_counters.qnn_graph_execute_us, value);
      break;
    case kAotDiagQnnSharedOutputCopybackUs:
      add_counter(g_aot_diag_counters.qnn_shared_output_copyback_us, value);
      break;
    case kAotDiagQnnGraphExecute:
      add_counter(g_aot_diag_counters.qnn_graph_execute_count, value);
      break;
    case kAotDiagQnnContextCreate:
      add_counter(g_aot_diag_counters.qnn_context_create_count, value);
      break;
    case kAotDiagQnnContextCreateFromBinary:
      add_counter(g_aot_diag_counters.qnn_context_create_from_binary_count, value);
      break;
    case kAotDiagQnnGraphFinalize:
      add_counter(g_aot_diag_counters.qnn_graph_finalize_count, value);
      break;
    case kAotDiagQnnCustomMemDtypeFloat32:
      add_counter(g_aot_diag_counters.qnn_custom_mem_dtype_float32_count, value);
      break;
    case kAotDiagQnnCustomMemDtypeFloat16:
      add_counter(g_aot_diag_counters.qnn_custom_mem_dtype_float16_count, value);
      break;
    case kAotDiagQnnCustomMemDtypeOther:
      add_counter(g_aot_diag_counters.qnn_custom_mem_dtype_other_count, value);
      break;
    case kAotDiagRpcmemAlloc:
      add_counter(g_aot_diag_counters.rpcmem_alloc_count, value);
      break;
    case kAotDiagRpcmemFree:
      add_counter(g_aot_diag_counters.rpcmem_free_count, value);
      break;
    case kAotDiagRpcmemTotalBytes:
      add_counter(g_aot_diag_counters.rpcmem_total_bytes, value);
      break;
    case kAotDiagCustomMemAddrMap:
      add_counter(g_aot_diag_counters.custom_mem_addr_map_count, value);
      break;
    case kAotDiagCustomMemAddrHit:
      add_counter(g_aot_diag_counters.custom_mem_addr_hit_count, value);
      break;
    case kAotDiagCustomMemAddrMiss:
      add_counter(g_aot_diag_counters.custom_mem_addr_miss_count, value);
      break;
    case kAotDiagQnnPmdHiddenSlotBypass:
      add_counter(g_aot_diag_counters.qnn_pmd_hidden_slot_bypass_count, value);
      break;
    case kAotDiagQnnSharedOutputCopyback:
      add_counter(g_aot_diag_counters.qnn_shared_output_copyback_count, value);
      break;
    case kAotDiagQnnSharedOutputCopybackBytes:
      add_counter(g_aot_diag_counters.qnn_shared_output_copyback_bytes, value);
      break;
    default:
      break;
  }
}

void QnnExecuTorchAotDiagGet(QnnExecuTorchAotDiagCounters* counters) {
  if (counters == nullptr) {
    return;
  }
  counters->qnn_backend_execute_count =
      load_counter(g_aot_diag_counters.qnn_backend_execute_count);
  counters->qnn_register_mem_check_count =
      load_counter(g_aot_diag_counters.qnn_register_mem_check_count);
  counters->qnn_register_mem_fallback_count =
      load_counter(g_aot_diag_counters.qnn_register_mem_fallback_count);
  counters->qnn_register_mem_custom_base_hit_count =
      load_counter(g_aot_diag_counters.qnn_register_mem_custom_base_hit_count);
  counters->qnn_register_mem_custom_base_miss_count =
      load_counter(g_aot_diag_counters.qnn_register_mem_custom_base_miss_count);
  counters->qnn_register_ion_attempt_count =
      load_counter(g_aot_diag_counters.qnn_register_ion_attempt_count);
  counters->qnn_register_custom_mem_entry_count =
      load_counter(g_aot_diag_counters.qnn_register_custom_mem_entry_count);
  counters->qnn_shared_buffer_current_hit_count =
      load_counter(g_aot_diag_counters.qnn_shared_buffer_current_hit_count);
  counters->qnn_shared_buffer_preregistered_hit_count =
      load_counter(g_aot_diag_counters.qnn_shared_buffer_preregistered_hit_count);
  counters->qnn_mem_register_count =
      load_counter(g_aot_diag_counters.qnn_mem_register_count);
  counters->qnn_ion_mem_register_count =
      load_counter(g_aot_diag_counters.qnn_ion_mem_register_count);
  counters->qnn_custom_mem_register_count =
      load_counter(g_aot_diag_counters.qnn_custom_mem_register_count);
  counters->qnn_set_mem_handle_count =
      load_counter(g_aot_diag_counters.qnn_set_mem_handle_count);
  counters->qnn_fill_data_buffer_count =
      load_counter(g_aot_diag_counters.qnn_fill_data_buffer_count);
  counters->qnn_fill_data_buffer_bytes =
      load_counter(g_aot_diag_counters.qnn_fill_data_buffer_bytes);
  counters->qnn_backend_execute_us =
      load_counter(g_aot_diag_counters.qnn_backend_execute_us);
  counters->qnn_prepare_inputs_us =
      load_counter(g_aot_diag_counters.qnn_prepare_inputs_us);
  counters->qnn_prepare_outputs_us =
      load_counter(g_aot_diag_counters.qnn_prepare_outputs_us);
  counters->qnn_graph_execute_us =
      load_counter(g_aot_diag_counters.qnn_graph_execute_us);
  counters->qnn_shared_output_copyback_us =
      load_counter(g_aot_diag_counters.qnn_shared_output_copyback_us);
  counters->qnn_graph_execute_count =
      load_counter(g_aot_diag_counters.qnn_graph_execute_count);
  counters->qnn_context_create_count =
      load_counter(g_aot_diag_counters.qnn_context_create_count);
  counters->qnn_context_create_from_binary_count =
      load_counter(g_aot_diag_counters.qnn_context_create_from_binary_count);
  counters->qnn_graph_finalize_count =
      load_counter(g_aot_diag_counters.qnn_graph_finalize_count);
  counters->qnn_custom_mem_dtype_float32_count =
      load_counter(g_aot_diag_counters.qnn_custom_mem_dtype_float32_count);
  counters->qnn_custom_mem_dtype_float16_count =
      load_counter(g_aot_diag_counters.qnn_custom_mem_dtype_float16_count);
  counters->qnn_custom_mem_dtype_other_count =
      load_counter(g_aot_diag_counters.qnn_custom_mem_dtype_other_count);
  counters->rpcmem_alloc_count =
      load_counter(g_aot_diag_counters.rpcmem_alloc_count);
  counters->rpcmem_free_count =
      load_counter(g_aot_diag_counters.rpcmem_free_count);
  counters->rpcmem_total_bytes =
      load_counter(g_aot_diag_counters.rpcmem_total_bytes);
  counters->custom_mem_addr_map_count =
      load_counter(g_aot_diag_counters.custom_mem_addr_map_count);
  counters->custom_mem_addr_hit_count =
      load_counter(g_aot_diag_counters.custom_mem_addr_hit_count);
  counters->custom_mem_addr_miss_count =
      load_counter(g_aot_diag_counters.custom_mem_addr_miss_count);
  counters->qnn_pmd_hidden_slot_bypass_count =
      load_counter(g_aot_diag_counters.qnn_pmd_hidden_slot_bypass_count);
  counters->qnn_shared_output_copyback_count =
      load_counter(g_aot_diag_counters.qnn_shared_output_copyback_count);
  counters->qnn_shared_output_copyback_bytes =
      load_counter(g_aot_diag_counters.qnn_shared_output_copyback_bytes);
}
