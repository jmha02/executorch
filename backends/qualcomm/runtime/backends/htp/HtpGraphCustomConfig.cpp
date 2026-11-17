/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCache.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpGraphCustomConfig.h>
namespace executorch {
namespace backends {
namespace qnn {
std::vector<QnnGraph_CustomConfig_t>
HtpGraphCustomConfig::CreateGraphCustomConfigCommon(
    const SocInfo* qcom_target_soc_info,
    float opt_level) {
  std::vector<QnnGraph_CustomConfig_t> ret;
  QnnHtpGraph_CustomConfig_t* p_custom_config = nullptr;

  if (!htp_options_->use_conv_hmx()) {
    p_custom_config = AllocGraphCustomConfig();
    p_custom_config->option =
        QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF;
    p_custom_config->shortDepthConvOnHmxOff = true;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));
  }

  if (!htp_options_->use_fold_relu()) {
    p_custom_config = AllocGraphCustomConfig();
    p_custom_config->option =
        QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
    p_custom_config->foldReluActivationIntoConvOff = true;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));
  }

  switch (htp_options_->precision()) {
    case QnnExecuTorchHtpPrecision::kHtpFp16:
      p_custom_config = AllocGraphCustomConfig();
      p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      p_custom_config->precision = QNN_PRECISION_FLOAT16;
      ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));
      break;
    case QnnExecuTorchHtpPrecision::kHtpQuantized:
    default:
      break;
  }

  if (htp_options_->optimization_level() >= 0) {
    opt_level = htp_options_->optimization_level();
  }
  QNN_EXECUTORCH_LOG_INFO(
      "Running level=%d optimization.", static_cast<int>(opt_level));

  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  p_custom_config->optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  p_custom_config->optimizationOption.floatValue = opt_level;
  ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  p_custom_config->vtcmSizeInMB = htp_options_->vtcm_size_in_mb() >= 0
      ? htp_options_->vtcm_size_in_mb()
      : qcom_target_soc_info->htp_info()->vtcm_size_in_mb();
  ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  p_custom_config->optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
  p_custom_config->optimizationOption.floatValue =
      htp_options_->use_dlbc() ? 1.0 : 0.0;
  ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  p_custom_config->optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_SLC_ALLOCATOR;
  p_custom_config->optimizationOption.floatValue =
      htp_options_->use_slc_allocator() ? 1.0 : 0.0;
  ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  auto add = [&](QnnHtpGraph_ConfigOption_t option) {
    p_custom_config = AllocGraphCustomConfig();
    p_custom_config->option = option;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));
    return p_custom_config;
  };
  if (htp_options_->finalize_p() >= 0) {
    p_custom_config = add(QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG);
    p_custom_config->finalizeConfig.key = "P";
    p_custom_config->finalizeConfig.value.dataType = QNN_DATATYPE_UINT_32;
    p_custom_config->finalizeConfig.value.uint32Value =
        htp_options_->finalize_p();
  }
  if (htp_options_->num_hvx_threads() >= 0) {
    add(QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS)->numHvxThreads =
        htp_options_->num_hvx_threads();
  }
  if (htp_options_->advanced_activation_fusion() >= 0) {
    add(QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION)
        ->advancedActivationFusion = htp_options_->advanced_activation_fusion();
  }
  if (htp_options_->high_precision_sigmoid() >= 0) {
    add(QNN_HTP_GRAPH_CONFIG_OPTION_HIGH_PRECISION_SIGMOID)
        ->highPrecisionSigmoid = htp_options_->high_precision_sigmoid();
  }
  if (htp_options_->weights_packing() >= 0) {
    add(QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING)->weightsPacking =
        htp_options_->weights_packing();
  }
  if (htp_options_->dlbc_weights() >= 0) {
    p_custom_config = add(QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION);
    p_custom_config->optimizationOption.type =
        QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC_WEIGHTS;
    p_custom_config->optimizationOption.floatValue =
        htp_options_->dlbc_weights();
  }

  return ret;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
