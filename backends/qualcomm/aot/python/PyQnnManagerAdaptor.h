/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <executorch/backends/qualcomm/aot/wrappers/OpWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace py = pybind11;
namespace executorch {
namespace backends {
namespace qnn {
class PyQnnOpWrapper {
 public:
  explicit PyQnnOpWrapper(
      const std::string& name,
      const std::string& package_name,
      const std::string& op_type) {
    op_wrapper_ = std::make_shared<OpWrapper>(name, package_name, op_type);
  }
  void AddInputTensors(
      const std::vector<std::shared_ptr<TensorWrapper>>& tensors) {
    op_wrapper_->AddInputTensors(tensors);
  }

  void AddOutputTensors(
      const std::vector<std::shared_ptr<TensorWrapper>>& tensors) {
    op_wrapper_->AddOutputTensors(tensors);
  }

  void AddTensorParam(
      const std::string& name,
      Qnn_DataType_t data_type,
      std::uint32_t rank,
      const std::vector<uint32_t>& dims,
      py::array& data,
      bool copy_data) {
    op_wrapper_->AddTensorParam(
        name, data_type, rank, dims.data(), data.data(), copy_data);
  }

  void AddScalarParam(
      const std::string& name,
      Qnn_DataType_t data_type,
      py::dict& attrData) {
    switch (data_type) {
      case Qnn_DataType_t::QNN_DATATYPE_INT_32:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<int32_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_INT_16:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<int16_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_INT_8:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<int8_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_UINT_32:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<uint32_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_UINT_16:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<uint16_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_UINT_8:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<uint8_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_FLOAT_32:
      case Qnn_DataType_t::QNN_DATATYPE_FLOAT_16:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<float>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_BOOL_8:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<bool>());
        break;
      default:
        QNN_EXECUTORCH_LOG_ERROR(
            "%s has invalid data type: %d", name.c_str(), data_type);
        break;
    }
  }
  std::shared_ptr<OpWrapper>& GetOpWrapper() {
    return op_wrapper_;
  }

 private:
  std::shared_ptr<OpWrapper> op_wrapper_;
};

class PyQnnTensorWrapper {
 public:
  explicit PyQnnTensorWrapper(const std::shared_ptr<TensorWrapper>& wrapper) {
    tensor_wrapper_ = wrapper;
  }
  struct EncodingData {
    float scale;
    int32_t offset;
  };
  struct Encoding {
    py::array_t<EncodingData> data;
    int32_t axis;
  };

  py::array_t<std::uint32_t> GetDims() {
    std::uint32_t* dim = tensor_wrapper_->GetDims();
    size_t shape[1]{tensor_wrapper_->GetRank()};
    size_t stride[1]{sizeof(std::uint32_t)};
    auto ret = py::array_t<std::uint32_t>(shape, stride);
    auto view = ret.mutable_unchecked<1>();
    for (int i = 0; i < ret.shape(0); ++i) {
      view(i) = dim[i];
    }
    return ret;
  }
  std::string GetName() {
    return tensor_wrapper_->GetName();
  }
  Qnn_DataType_t GetDataType() {
    return tensor_wrapper_->GetDataType();
  }
  Encoding GetEncodings() {
    auto q_param = tensor_wrapper_->GetQuantizeParams();
    size_t stride[1]{sizeof(EncodingData)};

    switch (q_param.quantizationEncoding) {
      case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
        Qnn_ScaleOffset_t data = q_param.scaleOffsetEncoding;
        size_t shape[1]{1};
        auto enc_data = py::array_t<EncodingData>(shape, stride);
        auto view = enc_data.mutable_unchecked<1>();
        view(0) = {data.scale, data.offset};
        return {enc_data, -1};
      }
      case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
        Qnn_AxisScaleOffset_t data = q_param.axisScaleOffsetEncoding;
        size_t shape[1]{data.numScaleOffsets};
        auto enc_data = py::array_t<EncodingData>(shape, stride);
        auto view = enc_data.mutable_unchecked<1>();
        for (int i = 0; i < enc_data.shape(0); ++i) {
          view(i) = {data.scaleOffset[i].scale, data.scaleOffset[i].offset};
        }
        return {enc_data, data.axis};
      }
      case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET: {
        Qnn_BwScaleOffset_t data = q_param.bwScaleOffsetEncoding;
        size_t shape[1]{1};
        auto enc_data = py::array_t<EncodingData>(shape, stride);
        auto view = enc_data.mutable_unchecked<1>();
        view(0) = {data.scale, data.offset};
        return {enc_data, -1};
      }
      case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
        Qnn_BwAxisScaleOffset_t data = q_param.bwAxisScaleOffsetEncoding;
        size_t shape[1]{data.numElements};
        auto enc_data = py::array_t<EncodingData>(shape, stride);
        auto view = enc_data.mutable_unchecked<1>();
        for (int i = 0; i < enc_data.shape(0); ++i) {
          view(i) = {data.scales[i], data.offsets[i]};
        }
        return {enc_data, data.axis};
      }
      default:
        QNN_EXECUTORCH_LOG_WARN(
            "%s QNN_QUANTIZATION_ENCODING_UNDEFINED detected",
            GetName().c_str());
        break;
    }
    return {};
  }

 private:
  std::shared_ptr<TensorWrapper> tensor_wrapper_;
};
class PyQnnManager {
 public:
  // used for AoT compilation
  explicit PyQnnManager(const py::bytes& buffer)
      : qnn_executorch_option_ptr_(buffer),
        qnn_executorch_context_binary_(QNN_EXECUTORCH_CONTEXT_BINARY) {
    // Choose non-allocating non-owning string pieces exposed as string_view for
    // parsers
    auto qnn_executorch_options = GetQnnExecuTorchOptions(
        qnn_executorch_option_ptr_.cast<std::string_view>().data());
    qnn_manager_ = std::make_shared<QnnManager>(
        qnn_executorch_options, qnn_executorch_context_binary_);
  }

  // used for loading context binary directly
  explicit PyQnnManager(const py::bytes& buffer, const py::bytes& ctx_bin)
      : qnn_executorch_option_ptr_(buffer) {
    auto qnn_executorch_options = GetQnnExecuTorchOptions(
        qnn_executorch_option_ptr_.cast<std::string_view>().data());

    py::buffer_info info(py::buffer(ctx_bin).request());
    qnn_executorch_context_binary_.buffer = info.ptr;
    qnn_executorch_context_binary_.nbytes = info.size * info.itemsize;
    qnn_manager_ = std::make_shared<QnnManager>(
        qnn_executorch_options, qnn_executorch_context_binary_);
  }

  executorch::runtime::Error Init() {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager_->InitBackend() == Error::Ok,
        Internal,
        "Fail to initailize backend");
    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager_->InitContext() == Error::Ok,
        Internal,
        "Fail to initailize context");
    return Error::Ok;
  }

  executorch::runtime::Error InitBackend() {
    return qnn_manager_->InitBackend();
  }

  executorch::runtime::Error InitContext(
      const std::vector<std::string>& graph_names) {
    return qnn_manager_->InitContext(std::optional{graph_names});
  }

  executorch::runtime::Error InitContextCache() {
    return qnn_manager_->InitContextCache();
  }

  bool IsNodeSupportedByBackend(
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
    return qnn_manager_->IsNodeSupportedByBackend(op_wrappers);
  }

  py::array_t<char> Compile(
      const std::vector<std::string>& graph_names,
      std::vector<std::vector<std::shared_ptr<OpWrapper>>>& op_wrappers) {
    return CompileImpl(graph_names, op_wrappers, nullptr);
  }

  py::tuple CompileWithUpdatableWeightsSection(
      const std::vector<std::string>& graph_names,
      std::vector<std::vector<std::shared_ptr<OpWrapper>>>& op_wrappers) {
    std::vector<uint8_t> section;
    py::array_t<char> context = CompileImpl(graph_names, op_wrappers, &section);
    return py::make_tuple(
        context,
        py::bytes(reinterpret_cast<const char*>(section.data()), section.size()));
  }

  py::tuple CompileWithUpdatedUpdatableWeightsSection(
      const std::vector<std::string>& graph_names,
      std::vector<std::vector<std::shared_ptr<OpWrapper>>>& op_wrappers) {
    std::vector<uint8_t> section;
    std::vector<uint8_t> base_context;
    py::array_t<char> context =
        CompileImpl(graph_names, op_wrappers, &section, true, &base_context);
    return py::make_tuple(
        context,
        py::bytes(
            reinterpret_cast<const char*>(base_context.data()),
            base_context.size()),
        py::bytes(reinterpret_cast<const char*>(section.data()), section.size()),
        GetUpdatableWeightsLifecycleTrace());
  }

  py::tuple CompileWithAllUpdatedUpdatableWeightsSection(
      const std::vector<std::string>& graph_names,
      std::vector<std::vector<std::shared_ptr<OpWrapper>>>& op_wrappers) {
    std::vector<uint8_t> section;
    std::vector<uint8_t> base_context;
    py::array_t<char> context = CompileImpl(
        graph_names, op_wrappers, &section, true, &base_context, true);
    return py::make_tuple(
        context,
        py::bytes(
            reinterpret_cast<const char*>(base_context.data()),
            base_context.size()),
        py::bytes(reinterpret_cast<const char*>(section.data()), section.size()),
        GetUpdatableWeightsLifecycleTrace());
  }

  void Destroy() {
    return qnn_manager_->Destroy();
  }

  void DestroyContext() {
    return qnn_manager_->DestroyContext();
  }

  bool IsAvailable() {
    return qnn_manager_->IsAvailable();
  }

  bool IsTensorDump() {
    return qnn_manager_->IsTensorDump();
  }

  executorch::runtime::Error AllocateTensor(const std::string& graph_name) {
    return qnn_manager_->AllocateTensor(graph_name);
  }

  py::list GetGraphInputs(const std::string& graph_name) {
    py::list ret;
    for (const std::shared_ptr<TensorWrapper>& input :
         qnn_manager_->GetGraphInputs(graph_name)) {
      ret.append(PyQnnTensorWrapper(input));
    }
    return ret;
  }

  py::list GetGraphOutputs(const std::string& graph_name) {
    py::list ret;
    for (const std::shared_ptr<TensorWrapper>& output :
         qnn_manager_->GetGraphOutputs(graph_name)) {
      ret.append(PyQnnTensorWrapper(output));
    }
    return ret;
  }

  py::list GetGraphNames() {
    py::list ret;
    for (const std::string& graph_name : qnn_manager_->GetGraphNames()) {
      ret.append(graph_name);
    }
    return ret;
  }

  uint64_t GetSpillFillBufferSize() {
    return qnn_manager_->GetSpillFillBufferSize();
  }

  py::array_t<char> MakeBinaryInfo(const py::bytes& ctx_bin) {
    py::buffer_info info(py::buffer(ctx_bin).request());
    QnnExecuTorchContextBinary binary(
        {info.ptr, static_cast<uint64_t>(info.size * info.itemsize)});

    auto qnn_context_custom_protocol = QnnContextCustomProtocol(binary.nbytes);
    qnn_context_custom_protocol.BuildContextCustomBuffer(binary);
    auto [custom_buffer_ptr, custom_buffer_size] =
        qnn_context_custom_protocol.GetCustomProtocolBuffer();

    auto result = py::array_t<char>(custom_buffer_size);
    auto result_buffer = result.request();
    std::memcpy(result_buffer.ptr, custom_buffer_ptr, custom_buffer_size);
    return result;
  }

  py::bytes GetUpdatableWeightsBinarySection(const std::string& graph_name) {
    std::vector<uint8_t> section;
    if (qnn_manager_->GetUpdatableWeightsBinarySection(graph_name, section) !=
        executorch::runtime::Error::Ok) {
      throw std::runtime_error(
          "failed to extract QNN_CONTEXT_SECTION_UPDATABLE_WEIGHTS");
    }
    return py::bytes(
        reinterpret_cast<const char*>(section.data()), section.size());
  }

  py::dict GetUpdatableWeightsLifecycleTrace() const {
    const auto& trace = qnn_manager_->GetUpdatableWeightsLifecycleTrace();
    py::dict result;
    result["tensor_update_graph_tensors_available"] =
        trace.tensor_update_graph_tensors_available;
    result["graph_finalize_available"] = trace.graph_finalize_available;
    result["binary_section_weights_updates_config_enabled"] =
        trace.binary_section_weights_updates_config_enabled;
    result["base_context_serialization_status"] =
        trace.base_context_serialization_status;
    result["base_context_bytes"] = trace.base_context_bytes;
    result["base_context_fnv1a64"] = trace.base_context_fnv1a64;
    result["tensor_update_status"] = trace.tensor_update_status;
    result["graph_refinalize_status"] = trace.graph_refinalize_status;
    result["section_size_status"] = trace.section_size_status;
    result["section_extraction_status"] = trace.section_extraction_status;
    result["section_bytes"] = trace.section_bytes;
    result["tensor_name"] = trace.tensor_name;
    result["tensor_id"] = trace.tensor_id;
    result["tensor_dims"] = trace.tensor_dims;
    result["payload_bytes"] = trace.payload_bytes;
    result["payload_fnv1a64"] = trace.payload_fnv1a64;
    py::list updated_tensors;
    for (const auto& tensor : trace.updated_tensors) {
      py::dict tensor_result;
      tensor_result["tensor_name"] = tensor.tensor_name;
      tensor_result["tensor_id"] = tensor.tensor_id;
      tensor_result["tensor_dims"] = tensor.tensor_dims;
      tensor_result["payload_bytes"] = tensor.payload_bytes;
      tensor_result["payload_fnv1a64"] = tensor.payload_fnv1a64;
      updated_tensors.append(tensor_result);
    }
    result["updated_tensors"] = updated_tensors;
    return result;
  }

 private:
  py::array_t<char> CompileImpl(
      const std::vector<std::string>& graph_names,
      std::vector<std::vector<std::shared_ptr<OpWrapper>>>& op_wrappers,
      std::vector<uint8_t>* section,
      bool update_and_refinalize = false,
      std::vector<uint8_t>* base_context = nullptr,
      bool update_all = false) {
    QnnExecuTorchContextBinary binary_info;
    qnn_manager_->ResetUpdatableWeightsLifecycleTrace();

    for (uint32_t i = 0; i < graph_names.size(); ++i) {
      if (qnn_manager_->Compile(graph_names[i], op_wrappers[i]) !=
          executorch::runtime::Error::Ok) {
        QNN_EXECUTORCH_LOG_ERROR("Fail to compile QNN graph");
        return py::array_t<char>(0);
      }
    }
    if (update_and_refinalize) {
      QnnExecuTorchContextBinary base_binary;
      const executorch::runtime::Error base_status =
          qnn_manager_->GetContextBinary(base_binary);
      const uint64_t base_context_bytes =
          base_status == executorch::runtime::Error::Ok ? base_binary.nbytes : 0;
      uint64_t base_context_fnv1a64 = 1469598103934665603ULL;
      if (base_status == executorch::runtime::Error::Ok) {
        const auto* base_data = reinterpret_cast<const uint8_t*>(base_binary.buffer);
        for (uint64_t index = 0; index < base_binary.nbytes; ++index) {
          base_context_fnv1a64 ^= base_data[index];
          base_context_fnv1a64 *= 1099511628211ULL;
        }
      }
      qnn_manager_->RecordBaseContextSerialization(
          base_status, base_context_bytes, base_context_fnv1a64);
      if (base_status != executorch::runtime::Error::Ok) {
        return py::array_t<char>(0);
      }
      if (base_context != nullptr) {
        base_context->assign(
            reinterpret_cast<const uint8_t*>(base_binary.buffer),
            reinterpret_cast<const uint8_t*>(base_binary.buffer) +
                base_binary.nbytes);
      }
      if (graph_names.empty()) {
        return py::array_t<char>(0);
      }
      const executorch::runtime::Error update_status = update_all
          ? qnn_manager_->UpdateAllUpdatableStaticTensorsAndRefinalize(
                graph_names.front())
          : qnn_manager_->UpdateFirstUpdatableStaticTensorAndRefinalize(
                graph_names.front());
      if (update_status != executorch::runtime::Error::Ok) {
        return py::array_t<char>(0);
      }
    }
    if (section != nullptr) {
      if (graph_names.empty() ||
          qnn_manager_->GetUpdatableWeightsBinarySection(
              graph_names.front(), *section) != executorch::runtime::Error::Ok) {
        if (update_and_refinalize) {
          return py::array_t<char>(0);
        }
        throw std::runtime_error(
            "failed to extract QNN_CONTEXT_SECTION_UPDATABLE_WEIGHTS");
      }
    }
    auto qnn_executorch_options = GetQnnExecuTorchOptions(
        qnn_executorch_option_ptr_.cast<std::string_view>().data());
    if (qnn_executorch_options->saver() ||
        qnn_manager_->GetContextBinary(binary_info) !=
            executorch::runtime::Error::Ok) {
      return py::array_t<char>(0);
    }

    // allocate py::array (to pass the result of the C++ function to Python)
    auto result = py::array_t<char>(binary_info.nbytes);
    auto result_buffer = result.request();
    char* result_ptr = (char*)result_buffer.ptr;
    std::memcpy(result_ptr, binary_info.buffer, binary_info.nbytes);
    return result;
  }
  // Store the bytes object instead of a raw pointer so that this module will
  // keep the bytes alive.
  const py::bytes qnn_executorch_option_ptr_;
  QnnExecuTorchContextBinary qnn_executorch_context_binary_;
  std::shared_ptr<QnnManager> qnn_manager_;
  QnnContextCustomProtocol custom_context_custom_buffer_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
