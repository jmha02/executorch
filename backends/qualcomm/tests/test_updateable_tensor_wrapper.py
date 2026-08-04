import numpy as np
from pathlib import Path

import PyQnnManagerAdaptor as qnn


def test_updateable_static_tensor_wrapper_keeps_initialized_payload() -> None:
    payload = np.array([3, 7, 11, 13], dtype=np.uint8)

    wrapper = qnn.TensorWrapper(
        "adapter",
        qnn.QNN_TENSOR_TYPE_UPDATEABLE_STATIC,
        qnn.QNN_DATATYPE_UINT_8,
        qnn.QNN_QUANTIZATION_ENCODING_UNDEFINED,
        {},
        1,
        [payload.size],
        [],
        payload,
        True,
    )

    assert wrapper.HasInitialPayload()
    assert wrapper.GetInitialPayloadSize() == payload.nbytes


def test_updateable_static_tensor_wrapper_rejects_missing_payload() -> None:
    with np.testing.assert_raises(ValueError):
        qnn.TensorWrapper(
            "adapter",
            qnn.QNN_TENSOR_TYPE_UPDATEABLE_STATIC,
            qnn.QNN_DATATYPE_UINT_8,
            qnn.QNN_QUANTIZATION_ENCODING_UNDEFINED,
            {},
            1,
            [4],
            [],
            np.array([], dtype=np.uint8),
            False,
        )


def test_app_write_tensor_wrapper_does_not_retain_payload() -> None:
    payload = np.array([3, 7, 11, 13], dtype=np.uint8)

    wrapper = qnn.TensorWrapper(
        "input",
        qnn.QNN_TENSOR_TYPE_APP_WRITE,
        qnn.QNN_DATATYPE_UINT_8,
        qnn.QNN_QUANTIZATION_ENCODING_UNDEFINED,
        {},
        1,
        [payload.size],
        [],
        payload,
        True,
    )

    assert not wrapper.HasInitialPayload()
    assert wrapper.GetInitialPayloadSize() == payload.nbytes


def test_updateable_section_lifecycle_exposes_update_then_refinalize_surface() -> None:
    root = Path(__file__).parents[1]
    interface = (root / "runtime/backends/QnnFunctionInterface.h").read_text(
        encoding="utf-8"
    )
    manager = (root / "runtime/QnnManager.cpp").read_text(encoding="utf-8")

    assert "tensorUpdateGraphTensors" in interface
    assert "HasTensorUpdateGraphTensors" in interface
    update_offset = manager.index("qnn_tensor_update_graph_tensors")
    refinalize_offset = manager.index("QNN updateable graph re-finalize")
    assert update_offset < refinalize_offset
    adaptor = (root / "aot/python/PyQnnManagerAdaptor.h").read_text(
        encoding="utf-8"
    )
    lifecycle_failure = adaptor.index("UpdateFirstUpdatableStaticTensorAndRefinalize")
    assert "return py::array_t<char>(0);" in adaptor[lifecycle_failure:]


def test_updateable_section_lifecycle_serializes_base_before_update() -> None:
    root = Path(__file__).parents[1]
    adaptor = (root / "aot/python/PyQnnManagerAdaptor.h").read_text(
        encoding="utf-8"
    )

    base_serialization = adaptor.index("qnn_manager_->GetContextBinary(base_binary)")
    update = adaptor.index(
        "qnn_manager_->UpdateFirstUpdatableStaticTensorAndRefinalize"
    )

    assert base_serialization < update
    assert "base_context_serialization_status" in adaptor
    assert "base_context_bytes" in adaptor
    assert "section_extraction_status" in adaptor


def test_updateable_section_lifecycle_exposes_batched_lora_update_surface() -> None:
    root = Path(__file__).parents[1]
    manager = (root / "runtime/QnnManager.cpp").read_text(encoding="utf-8")
    adaptor = (root / "aot/python/PyQnnManagerAdaptor.h").read_text(
        encoding="utf-8"
    )

    assert "UpdateAllUpdatableStaticTensorsAndRefinalize" in manager
    assert "updated_tensors" in manager
    assert "CompileWithAllUpdatedUpdatableWeightsSection" in adaptor
    assert "UpdateAllUpdatableStaticTensorsAndRefinalize" in adaptor


def test_updateable_section_probe_can_omit_only_config7() -> None:
    root = Path(__file__).parents[1]
    manager = (root / "runtime/QnnManager.cpp").read_text(encoding="utf-8")

    assert "EXECUTORCH_QNN_OMIT_BINARY_SECTION_WEIGHTS_UPDATES_CONFIG" in manager
    assert "EnableBinarySectionWeightUpdates(graph_name)" in manager
    assert "binary_section_weights_updates_config_enabled" in manager
