# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import hashlib
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Collection, Dict, final, List

import torch  # noqa: F401
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.partition.utils import generate_qnn_executorch_option
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnLoRAPreparationRole,
    QnnExecuTorchOpPackageInfo,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER
from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import (
    get_current_qnn_manager,
)
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY
from torch.export.exported_program import ExportedProgram

DEFAULT_DEBUG_HANDLE = 65535
DEFAULT_GRAPH_NAME = "forward"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def validate_native_lora_preparation_records(
    records,
    *,
    expected_graph_identities: Collection[str],
) -> None:
    groups = defaultdict(set)
    parameter_groups = {}
    expected_graph_identity_set = frozenset(expected_graph_identities)
    for record in records:
        if not record.parameter_fqn or not record.group_id or not record.graph_identity:
            raise RuntimeError("native LoRA preparation record has incomplete identity")
        if record.graph_identity not in expected_graph_identity_set:
            raise RuntimeError(
                "native LoRA preparation record has foreign graph identity: "
                f"{record.graph_identity}"
            )
        if not record.dimensions or not record.dtype or record.layout != "row_major":
            raise RuntimeError("native LoRA preparation record has invalid tensor metadata")
        if any(dimension <= 0 for dimension in record.dimensions):
            raise RuntimeError(
                "native LoRA preparation record has non-positive dimension: "
                f"{record.parameter_fqn}"
            )
        prior_group = parameter_groups.setdefault(record.parameter_fqn, record.group_id)
        if prior_group != record.group_id:
            raise RuntimeError(
                "native LoRA preparation record duplicates parameter FQN across groups: "
                f"{record.parameter_fqn}"
            )
        if record.role in groups[record.group_id]:
            raise RuntimeError(f"native LoRA preparation record duplicates role in {record.group_id}")
        groups[record.group_id].add(record.role)
    expected = {QnnLoRAPreparationRole.A, QnnLoRAPreparationRole.B}
    for group_id, roles in groups.items():
        if roles != expected:
            raise RuntimeError(f"native LoRA preparation record is incomplete for {group_id}")


def compile_with_optional_updatable_weights_section(
    qnn_manager, graph_names, op_wrappers
):
    capture_path = os.environ.get("EXECUTORCH_QNN_UPDATABLE_WEIGHTS_SECTION_PATH")
    if capture_path is None:
        return qnn_manager.Compile(graph_names, op_wrappers)
    refinalize = os.environ.get(
        "EXECUTORCH_QNN_UPDATABLE_WEIGHTS_REFINALIZE", ""
    ) == "1"
    update_all = os.environ.get(
        "EXECUTORCH_QNN_UPDATABLE_WEIGHTS_UPDATE_ALL", ""
    ) == "1"
    if refinalize:
        if update_all:
            context, base_context, section, trace = (
                qnn_manager.CompileWithAllUpdatedUpdatableWeightsSection(
                    graph_names, op_wrappers
                )
            )
        else:
            context, base_context, section, trace = (
                qnn_manager.CompileWithUpdatedUpdatableWeightsSection(
                    graph_names, op_wrappers
                )
            )
    else:
        context, section = qnn_manager.CompileWithUpdatableWeightsSection(
            graph_names, op_wrappers
        )
    target = Path(capture_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(section))
    if refinalize:
        base_context_path = target.with_suffix(target.suffix + ".base_context.bin")
        base_context_bytes = bytes(base_context)
        base_context_path.write_bytes(base_context_bytes)
        trace["base_context_path"] = str(base_context_path.resolve())
        trace["base_context_sha256"] = hashlib.sha256(base_context_bytes).hexdigest()
        target.with_suffix(target.suffix + ".trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8"
        )
    return context


@final
class QnnBackend(BackendDetails):
    @staticmethod
    def _build_op_wrappers(
        edge_program: ExportedProgram,
        enable_tensor_dump: bool,
        op_package_infos: List[QnnExecuTorchOpPackageInfo],
        use_mha2sha: bool,
    ):
        for node in edge_program.graph_module.graph.nodes:
            if hasattr(node, "meta"):
                # pop certain keys in meta for not affecting the passes in compilation
                node.meta.pop(QCOM_AXIS_ORDER, "")
        # QNN Delegate Specific Passes
        graph_module = QnnPassManager().transform_for_preprocess_pipeline(
            edge_program, use_mha2sha=use_mha2sha
        )
        assert graph_module is not None

        nodes_to_wrappers = defaultdict(dict)
        node_visitors = get_node_visitors(
            edge_program,
            enable_tensor_dump=enable_tensor_dump,
            op_package_infos=op_package_infos,
        )
        py_op_wrapper_list = []
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                logger.info(f"Visiting: {node}, {node.target.__name__}")
                if node.target.__name__ in node_visitors:
                    py_op_wrapper = node_visitors[node.target.__name__].define_node(
                        node, nodes_to_wrappers
                    )
                    if py_op_wrapper is not None:
                        if isinstance(py_op_wrapper, List):
                            py_op_wrapper_list.extend(py_op_wrapper)
                        else:
                            py_op_wrapper_list.append(py_op_wrapper)
                else:
                    err_msg = (
                        f"For {node}, {node.op}:{node.target.__name__} "
                        "is not supported in Qnn Delegate"
                    )
                    try:
                        context_loader_target = eval(
                            f"torch.ops.{OpContextLoader.namespace}.{node.target.__name__}",
                            globals().update(torch.__dict__),
                        )
                        assert node.target == context_loader_target, err_msg
                        # if graph has context binary loader node, return directly
                        return node.meta[OpContextLoader.meta_ctx_bin]
                    except:
                        raise RuntimeError(err_msg)

            elif node.op in [
                "get_attr",
                "placeholder",
                "output",
            ]:
                continue
            else:
                raise RuntimeError(f"{node.op} is not supported in Qnn")

        return py_op_wrapper_list

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        option = generate_qnn_executorch_option(compile_specs)
        obj_options = flatbuffer_to_option(option)
        validate_native_lora_preparation_records(
            obj_options.native_lora_preparation_records,
            expected_graph_identities={DEFAULT_GRAPH_NAME},
        )
        qnn_manager = get_current_qnn_manager(
            obj_options.backend_options.backend_type, compile_specs
        )
        qnn_manager.InitContext([DEFAULT_GRAPH_NAME])
        py_op_wrapper_list = QnnBackend._build_op_wrappers(
            edge_program,
            qnn_manager.IsTensorDump(),
            obj_options.op_package_options.op_package_infos,
            obj_options.use_mha2sha,
        )

        qnn_context_binary = compile_with_optional_updatable_weights_section(
            qnn_manager,
            qnn_manager.GetGraphNames(),
            [[py_op_wrapper.GetOpWrapper() for py_op_wrapper in py_op_wrapper_list]],
        )

        if obj_options.saver:
            exit(
                f"Record all QNN API calls from saver backend at: {obj_options.saver_output_dir}"
            )
        assert len(qnn_context_binary) != 0, "Failed to generate Qnn context binary."
        qnn_manager.DestroyContext()
        # For now, debug_handle_map is not used by QNN ExecuTorch
        return PreprocessResult(
            processed_bytes=bytes(qnn_context_binary),
            debug_handle_map={},
        )

    @staticmethod
    def preprocess_multimethod(  # noqa: C901
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> PreprocessResult:
        # TODO: refactor QnnManager to consume multiple compile_spec
        # take first compile_specs here for the same partitions
        graph_names = list(edge_programs.keys())
        compile_spec = list(compile_specs.values())[0][0]
        option = flatbuffer_to_option(compile_spec[0].value)
        validate_native_lora_preparation_records(
            option.native_lora_preparation_records,
            expected_graph_identities=set(graph_names),
        )
        # check if each graph has equal number of partitions
        num_sub_graphs = set()
        for edge_program in edge_programs.values():
            num_sub_graphs.add(len(edge_program))
        # this constraint is dedicated to weight-sharing scenario
        assert (
            len(num_sub_graphs) == 1
        ), "Only graphs with the same number of partitions could be used"

        all_processed_results = {key: [] for key in edge_programs.keys()}
        num_sub_graphs = next(iter(num_sub_graphs))
        qnn_manager = get_current_qnn_manager(
            option.backend_options.backend_type, compile_spec
        )
        debug_handle_builder = DelegateMappingBuilder(generated_identifiers=False)
        for i in range(num_sub_graphs):
            # e.g. 2 methods (x, y) with 3 subgraphs(partitions)
            #      > context_binary_0: [x.subgraph_0, y.subgraph_0]
            #      > context_binary_1: [x.subgraph_1, y.subgraph_1]
            #      > context_binary_2: [x.subgraph_2, y.subgraph_2]
            qnn_manager.InitContext(graph_names)
            py_op_wrapper_list, ctx_binary_list = [], []
            for j, programs in enumerate(edge_programs.values()):
                logger.info(f"Processing Method({j}): ({i+1}/{num_sub_graphs})")
                py_op_wrappers = QnnBackend._build_op_wrappers(
                    programs[i],
                    qnn_manager.IsTensorDump(),
                    option.op_package_options.op_package_infos,
                    option.use_mha2sha,
                )
                if qnn_manager.IsTensorDump():
                    for node in programs[i].graph.nodes:
                        if handle_id := node.meta.get(DEBUG_HANDLE_KEY):
                            debug_handle_builder.insert_delegate_mapping_entry(
                                handles=handle_id,
                                identifier=node.name,
                            )
                if isinstance(py_op_wrappers, bytes):
                    ctx_binary_list.append(py_op_wrappers)
                else:
                    py_op_wrapper_list.append(
                        [
                            py_op_wrapper.GetOpWrapper()
                            for py_op_wrapper in py_op_wrappers
                        ]
                    )

            if len(py_op_wrapper_list) == len(edge_programs.values()):
                qnn_context_binary = compile_with_optional_updatable_weights_section(
                    qnn_manager, graph_names, py_op_wrapper_list
                )
                if option.saver:
                    # TODO: Currently, only the first method is saved. Update this logic if saving multiple methods becomes necessary in the future.
                    exit(
                        f"Record all QNN API calls from saver backend at: {option.saver_output_dir}"
                    )
                assert (
                    len(qnn_context_binary) != 0
                ), "Failed to generate Qnn context binary."
                qnn_manager.DestroyContext()
                # methods should share the same context binary for current partition
                for key in edge_programs.keys():
                    all_processed_results[key].append(
                        PreprocessResult(
                            processed_bytes=bytes(qnn_context_binary),
                            debug_handle_map=debug_handle_builder.get_delegate_mapping(),
                        )
                    )
            elif len(ctx_binary_list) == len(edge_programs.values()):
                for i, key in enumerate(edge_programs.keys()):
                    all_processed_results[key].append(
                        PreprocessResult(
                            processed_bytes=ctx_binary_list[i],
                            debug_handle_map=debug_handle_builder.get_delegate_mapping(),
                        )
                    )
            else:
                raise RuntimeError("Hybrid compilation is not supported")

        return all_processed_results
