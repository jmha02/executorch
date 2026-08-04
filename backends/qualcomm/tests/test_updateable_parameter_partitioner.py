# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.utils.utils import to_edge_transform_and_lower_to_qnn
from executorch.exir.backend.utils import tag_constant_data
from torch.fx.passes.infra.partitioner import Partition


class TinyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.ones(2, 2))
        self.fixed = torch.nn.Parameter(torch.ones(2, 2))
        self.register_buffer("buffer", torch.ones(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.adapter + x @ self.fixed + x @ self.buffer


def exported_tiny_adapter() -> torch.export.ExportedProgram:
    return torch.export.export(TinyAdapter().eval(), (torch.ones(1, 2),), strict=True)


def partitioner_for_graph(
    *,
    mutable_parameter_names: tuple[str, ...] = (),
    updateable_parameter_names: tuple[str, ...] = (),
) -> QnnPartitioner:
    partitioner = object.__new__(QnnPartitioner)
    partitioner.delegation_spec = object()
    partitioner.partition_tags = {}
    partitioner.mutable_parameter_names = set(mutable_parameter_names)
    partitioner.updateable_parameter_names = set(updateable_parameter_names)
    return partitioner


def tag_tiny_adapter_partition(
    partitioner: QnnPartitioner, program: torch.export.ExportedProgram
) -> None:
    call_nodes = [node for node in program.graph_module.graph.nodes if node.op == "call_function"]
    partitioner.tag_nodes([Partition(id=0, nodes=call_nodes)], program)
    tag_constant_data(program)


def parameter_node(
    program: torch.export.ExportedProgram, parameter_name: str
) -> torch.fx.Node:
    for node in program.graph_module.graph.nodes:
        if program.graph_signature.inputs_to_parameters.get(node.name) == parameter_name:
            return node
    raise AssertionError(f"missing parameter node {parameter_name}")


def buffer_node(program: torch.export.ExportedProgram, buffer_name: str) -> torch.fx.Node:
    for node in program.graph_module.graph.nodes:
        if program.graph_signature.inputs_to_buffers.get(node.name) == buffer_name:
            return node
    raise AssertionError(f"missing buffer node {buffer_name}")


def test_mutable_parameter_remains_top_level_runtime_input() -> None:
    # Given: an actual torch.export graph with a delegated adapter parameter.
    program = exported_tiny_adapter()
    partitioner = partitioner_for_graph(mutable_parameter_names=("adapter",))
    tag_tiny_adapter_partition(partitioner, program)

    # When: the existing APP_WRITE lowering contract is applied.
    partitioner.untag_mutable_parameters(program)

    # Then: only the mutable adapter loses its delegate tag and becomes a runtime input.
    adapter = parameter_node(program, "adapter")
    fixed = parameter_node(program, "fixed")
    assert adapter.meta.get("delegation_tag") is None
    assert adapter.meta.get("qnn_mutable_parameter_input") is True
    assert fixed.meta.get("delegation_tag") == "qnn_0"


def test_updateable_parameter_keeps_delegation_tag() -> None:
    # Given: an actual delegated torch.export adapter parameter.
    program = exported_tiny_adapter()
    partitioner = partitioner_for_graph(updateable_parameter_names=("adapter",))
    tag_tiny_adapter_partition(partitioner, program)
    adapter = parameter_node(program, "adapter")
    tag_before = adapter.meta.get("delegation_tag")

    # When: the opt-in updateable lowering contract is applied.
    partitioner.mark_updateable_parameters(program)

    # Then: the selected parameter remains delegated and is marked for later tensor typing.
    assert tag_before == "qnn_0"
    assert adapter.meta.get("delegation_tag") == tag_before
    assert adapter.meta.get("qnn_updateable_parameter") is True


def test_updateable_marker_does_not_apply_to_buffers() -> None:
    # Given: a delegated graph whose selected name identifies a buffer, not a parameter.
    program = exported_tiny_adapter()
    partitioner = partitioner_for_graph(updateable_parameter_names=("buffer",))
    tag_tiny_adapter_partition(partitioner, program)

    # When: updateable parameter marking runs.
    partitioner.mark_updateable_parameters(program)

    # Then: the buffer remains a normal delegated tensor without the parameter marker.
    assert buffer_node(program, "buffer").meta.get("qnn_updateable_parameter") is None


def test_updateable_and_mutable_overlap_is_rejected_before_qnn_compilation() -> None:
    # Given: an adapter is requested through incompatible APP_WRITE and updateable paths.
    module = TinyAdapter().eval()

    # When / Then: the public lowering boundary rejects it before compiler specs are consumed.
    with pytest.raises(ValueError, match="mutable_parameter_names.*updateable_parameter_names.*adapter"):
        to_edge_transform_and_lower_to_qnn(
            module,
            (torch.ones(1, 2),),
            compiler_specs=[],
            mutable_parameter_names=("adapter",),
            updateable_parameter_names=("adapter",),
        )
