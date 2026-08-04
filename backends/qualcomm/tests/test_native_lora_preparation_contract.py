# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved

import pytest
import torch

from executorch.backends.qualcomm.utils.utils import (
    NativeLoRAPreparationError,
    build_native_lora_preparation_records,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendOptions,
    QnnExecuTorchBackendType,
    QnnExecuTorchOptions,
    SocInfo,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
    option_to_flatbuffer,
)
from executorch.backends.qualcomm.qnn_preprocess import (
    QnnBackend,
    validate_native_lora_preparation_records,
)


class TinyLoRA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.lora_A = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.float16))
        self.layer.lora_B = torch.nn.Parameter(torch.ones(8, 2, dtype=torch.float16))
        self.extra = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.layer.lora_A.transpose(0, 1) @ self.layer.lora_B.transpose(0, 1)


def test_build_native_lora_preparation_records_when_standard_pair_is_selected() -> None:
    # Given: a standard named LoRA A/B pair selected for the opt-in contract.
    module = TinyLoRA().eval()

    # When: export constructs compiler records before QNN compilation.
    records = build_native_lora_preparation_records(
        module,
        ("layer.lora_A", "layer.lora_B"),
        graph_identity="forward",
        scale_folded=True,
    )

    # Then: the complete typed group carries parameter, role, dimensions, and layout.
    assert [(record.parameter_fqn, record.role.name) for record in records] == [
        ("layer.lora_A", "A"),
        ("layer.lora_B", "B"),
    ]
    assert {record.group_id for record in records} == {"layer"}
    assert {record.graph_identity for record in records} == {"forward"}
    assert [record.dimensions for record in records] == [[2, 4], [8, 2]]
    assert {record.dtype for record in records} == {"float16"}
    assert {record.layout for record in records} == {"row_major"}
    assert all(record.scale_folded for record in records)


def test_native_lora_preparation_records_round_trip_into_qnn_preprocessing() -> None:
    # Given: a complete native preparation group selected from a standard LoRA module.
    records = build_native_lora_preparation_records(
        TinyLoRA().eval(),
        ("layer.lora_A", "layer.lora_B"),
        graph_identity="forward",
        scale_folded=False,
    )
    options = QnnExecuTorchOptions(
        soc_info=SocInfo(),
        backend_options=QnnExecuTorchBackendOptions(
            QnnExecuTorchBackendType.kUndefinedBackend
        ),
        native_lora_preparation_records=records,
    )

    # When: the compiler option is serialized then read by QNN preprocessing.
    decoded = flatbuffer_to_option(option_to_flatbuffer(options))
    validate_native_lora_preparation_records(
        decoded.native_lora_preparation_records,
        expected_graph_identities={"forward"},
    )

    # Then: the complete named group remains intact at the QNN boundary.
    assert [record.parameter_fqn for record in decoded.native_lora_preparation_records] == [
        "layer.lora_A",
        "layer.lora_B",
    ]
    assert [record.role.name for record in decoded.native_lora_preparation_records] == [
        "A",
        "B",
    ]


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        (("layer.lora_A",), "incomplete LoRA group"),
        (("layer.lora_A", "layer.lora_A", "layer.lora_B"), "duplicate LoRA role"),
        (("layer.lora_A", "layer.lora_B", "not_lora"), "recognized LoRA"),
    ],
)
def test_build_native_lora_preparation_records_rejects_malformed_selection(
    selected: tuple[str, ...], message: str
) -> None:
    # Given: a malformed or non-standard requested preparation selection.
    module = TinyLoRA().eval()

    # When / Then: the compiler boundary fails before QNN compilation.
    with pytest.raises(NativeLoRAPreparationError, match=message):
        build_native_lora_preparation_records(
            module, selected, graph_identity="forward", scale_folded=False
        )


def test_native_lora_preparation_rejects_mutable_overlap_before_qnn_compilation() -> None:
    # Given: a standard pair requested through both APP_WRITE and native preparation.
    module = TinyLoRA().eval()

    # When / Then: the public lowering boundary rejects the conflicting modes first.
    with pytest.raises(NativeLoRAPreparationError, match="overlaps mutable"):
        to_edge_transform_and_lower_to_qnn(
            module,
            (torch.ones(1, 4, dtype=torch.float16),),
            compiler_specs=[],
            mutable_parameter_names=["layer.lora_A", "layer.lora_B"],
            native_lora_preparation_parameter_names=["layer.lora_A", "layer.lora_B"],
        )


def test_app_write_remains_the_default_when_native_preparation_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a regular APP_WRITE request with no native preparation selection.
    observed: dict[str, object] = {}

    class RecordingPartitioner:
        def __init__(self, _compiler_specs, **kwargs) -> None:
            observed["mutable"] = kwargs["mutable_parameter_names"]
            observed["updateable"] = kwargs["updateable_parameter_names"]

    monkeypatch.setattr(
        "executorch.backends.qualcomm.utils.utils.QnnPartitioner",
        RecordingPartitioner,
    )
    monkeypatch.setattr(
        "executorch.backends.qualcomm.utils.utils.torch.export.export",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after partitioner")),
    )

    # When: lowering is invoked through the unchanged mutable-parameter API.
    with pytest.raises(RuntimeError, match="stop after partitioner"):
        to_edge_transform_and_lower_to_qnn(
            TinyLoRA().eval(),
            (torch.ones(1, 4, dtype=torch.float16),),
            compiler_specs=generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SM8550,
                backend_options=generate_htp_compiler_spec(use_fp16=True),
            ),
            mutable_parameter_names=["layer.lora_A", "layer.lora_B"],
        )

    # Then: APP_WRITE mutable inputs retain their previous partitioner contract;
    # no UPDATEABLE_STATIC promotion occurs without the native opt-in.
    assert observed["mutable"] == ["layer.lora_A", "layer.lora_B"]
    assert observed["updateable"] is None


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("duplicate_parameter_fqn", "duplicates parameter FQN"),
        ("nonpositive_dimension", "non-positive dimension"),
        ("foreign_graph_identity", "foreign graph identity"),
    ],
)
def test_qnn_preprocess_rejects_malformed_records_at_the_compiler_boundary(
    malformation: str,
    message: str,
) -> None:
    # Given: a serialized compiler option containing a complete but malformed
    # native preparation contract.
    records = build_native_lora_preparation_records(
        TinyLoRA().eval(),
        ("layer.lora_A", "layer.lora_B"),
        graph_identity="forward",
        scale_folded=False,
    )
    match malformation:
        case "duplicate_parameter_fqn":
            records[1].parameter_fqn = records[0].parameter_fqn
            records[1].group_id = "foreign_group"
        case "nonpositive_dimension":
            records[0].dimensions[0] = 0
        case "foreign_graph_identity":
            records[0].graph_identity = "foreign"
        case _:
            raise AssertionError(f"unhandled malformation: {malformation}")
    options = QnnExecuTorchOptions(
        soc_info=SocInfo(),
        backend_options=QnnExecuTorchBackendOptions(
            QnnExecuTorchBackendType.kUndefinedBackend
        ),
        native_lora_preparation_records=records,
    )
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8550,
        backend_options=generate_htp_compiler_spec(use_fp16=True),
    )
    compiler_specs[0].value = option_to_flatbuffer(options)

    # When / Then: QNN preprocessing rejects it before it can initialize a
    # manager or invoke the backend compiler.
    with pytest.raises(RuntimeError, match=message):
        QnnBackend.preprocess(None, compiler_specs)


def test_native_lora_preparation_promotes_only_its_typed_pair_to_updateable_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an opt-in native LoRA A/B contract with no APP_WRITE or explicit
    # updateable-static selection.
    observed: dict[str, object] = {}

    class RecordingPartitioner:
        def __init__(self, _compiler_specs, **kwargs) -> None:
            observed["mutable"] = kwargs["mutable_parameter_names"]
            observed["updateable"] = kwargs["updateable_parameter_names"]

    monkeypatch.setattr(
        "executorch.backends.qualcomm.utils.utils.QnnPartitioner",
        RecordingPartitioner,
    )
    monkeypatch.setattr(
        "executorch.backends.qualcomm.utils.utils.torch.export.export",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after partitioner")),
    )

    # When: lowering consumes the typed native preparation selection.
    with pytest.raises(RuntimeError, match="stop after partitioner"):
        to_edge_transform_and_lower_to_qnn(
            TinyLoRA().eval(),
            (torch.ones(1, 4, dtype=torch.float16),),
            compiler_specs=generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SM8550,
                backend_options=generate_htp_compiler_spec(use_fp16=True),
            ),
            native_lora_preparation_parameter_names=["layer.lora_A", "layer.lora_B"],
        )

    # Then: the sole public AOT representation is UPDATEABLE_STATIC, while
    # APP_WRITE remains absent from this opt-in path.
    assert observed["mutable"] is None
    assert observed["updateable"] == ["layer.lora_A", "layer.lora_B"]


def test_native_lora_preparation_rejects_nonoverlapping_explicit_updateable_parameter() -> None:
    # Given: an explicit updateable parameter outside the opt-in native A/B pair.
    module = TinyLoRA().eval()

    # When / Then: the precompile boundary rejects a request that would lose it.
    with pytest.raises(NativeLoRAPreparationError, match="cannot combine.*updateable"):
        to_edge_transform_and_lower_to_qnn(
            module,
            (torch.ones(1, 4, dtype=torch.float16),),
            compiler_specs=generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SM8550,
                backend_options=generate_htp_compiler_spec(use_fp16=True),
            ),
            updateable_parameter_names=["extra"],
            native_lora_preparation_parameter_names=["layer.lora_A", "layer.lora_B"],
        )
