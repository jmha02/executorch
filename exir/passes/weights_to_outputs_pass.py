# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Sequence, Set

from torch.export import ExportedProgram
from torch.export.exported_program import OutputKind, OutputSpec, TensorArgument


def weights_to_outputs_pass(
    exported_program: ExportedProgram,
    include_all_parameters: bool = False,
    parameter_names: Optional[Sequence[str]] = None,
) -> ExportedProgram:
    """
    This pass is for training graphs with gradients returned. It flags the weights as having a gradient attached,
    and appends them to the outputs in order to make the weights easier to handle in memory planning and the emitter.

    Args:
        exported_program: The ExportedProgram to update.

    Returns:
        The modified ExportedProgram.
    """
    if (
        len([node for node in exported_program.graph.nodes if node.op == "placeholder"])
        == 0
    ):
        return exported_program

    gs = exported_program.graph_signature
    gm = exported_program.graph_module

    # Check for/get gradients.
    grad_targets = [
        spec.target
        for spec in gs.output_specs
        if spec.kind == OutputKind.GRADIENT_TO_PARAMETER
    ]

    requested_names: Optional[Set[str]] = (
        set(parameter_names) if parameter_names is not None else None
    )

    # If no gradients and no forward-only mutable-weight request, return.
    if len(grad_targets) == 0 and not include_all_parameters and requested_names is None:
        return exported_program

    inputs_to_params = gs.inputs_to_parameters

    # Get output node
    output_node = gm.graph.output_node()

    if len(grad_targets) > 0:
        selected_param_names = set(grad_targets)
    elif requested_names is not None:
        selected_param_names = requested_names
    else:
        selected_param_names = set(inputs_to_params.values())

    # Get input nodes that should be exposed as mutable parameter tokens.
    placeholder_nodes = []
    for node in gm.graph.nodes:
        if node.op != "placeholder" or node.target not in inputs_to_params:
            continue
        if inputs_to_params[node.target] in selected_param_names:
            placeholder_nodes.append(node)

    if len(placeholder_nodes) == 0:
        return exported_program

    # Flag these placeholders so memory planning and mutable-weight externalization
    # treat them like runtime-updatable training weights.
    for node in placeholder_nodes:
        node.meta["weight_has_gradient"] = True
        node.meta["external_mutable_weight"] = True

    # add to output node
    new_output_nodes = []
    new_output_nodes.extend(output_node.args[0])
    new_output_nodes.extend(placeholder_nodes)
    # Remove old outputs
    new_output = gm.graph.output(tuple(new_output_nodes))
    output_node.replace_all_uses_with(new_output)
    gm.graph.erase_node(output_node)

    # add to output signature
    for node in placeholder_nodes:
        gs.output_specs.append(
            OutputSpec(
                OutputKind.TOKEN,  # This is a hack. We are returning the raw weights here to make it easier for memory
                # planning and the emitter. There is no outputkind.Parameter so I am using TOKEN which is currently unused in Edge.
                TensorArgument(node.target),
                None,
            )
        )

    # Cleanup the graph.
    exported_program.graph.eliminate_dead_code()
    exported_program.graph_module.recompile()

    return exported_program
